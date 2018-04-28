use std;    // Usaremos BTreeMap para ordenar los cromosomas en el procedimiento generacional
use knn::Dato;
use evaluacion_pesos::evaluar;
use ordered_float::OrderedFloat;
use rand::Rng;
use std::cell::RefCell;

use funciones_practica1::*;    // Recuperamos la búsqueda local y las funciones de combinación de algoritmos

type ArbolBinario = std::collections::BTreeMap<(OrderedFloat<f64>, usize), Vec<f64>>;



// Algunas constantes y funciones auxiliares


// Operador de cruce BLX-alfa con alfa = 0.3
// Recibe los padres y un RNG y devuelve dos hijos
fn blx_03<Trng: Rng>(padre1: &[f64], padre2: &[f64], rng: &mut Trng) -> Vec<Vec<f64>> {
    let mut hijos = Vec::new();
    for _i in 0..2 {
        let mut c_hijo = padre1.iter().zip(padre2.iter()).map(|(x, y)| {
            let x_menor = *x < *y;
            let (cmin, cmax) = if x_menor { (*x, *y) } else { (*y, *x) };
            let ai = (cmax - cmin)*0.3;
            let (a, b) = (cmin - ai, cmax + ai);
            let valor = if a == b {
                cmin
            } else {
                rng.gen_range(a, b)
            };
            // Devolvemos valor truncado a [0, 1]
            if valor <= 0.0 { 0.0 } else if valor >= 1.0 { 1.0 } else { valor }
        }).collect();
        normalizar(&mut c_hijo);  // Normalizamos el cromosoma para que el máximo sea 1
        hijos.push(c_hijo);
    }
    hijos
}


// Operador de cruce aritmético
// Recibe los padres y un RNG y devuelve un hijo
fn ca<Trng: Rng>(padre1: &[f64], padre2: &[f64], _rng: &mut Trng) -> Vec<Vec<f64>> {
    let mut unico_hijo = padre1.iter().zip(padre2.iter()).map(|(x, y)| (*x + *y)/2.0).collect();
    normalizar(&mut unico_hijo);  // Normalizamos el vector, porque seguramente el máximo ha dejado de ser 1
    vec![unico_hijo]
}

const TAMANO_GENERACIONAL: usize = 30;
const TAMANO_ESTACIONARIO: usize = 30;
const TASA_CRUCE_GENERACIONAL: f64 = 0.7;
const TASA_MUTACION_GEN: f64 = 0.001;
const MAX_EVALUACIONES: usize = 15000;
const MAX_CICLOS_BL: usize = 100000000; // No hay límite de ciclos, sino de evaluaciones


// Procedimiento memético consistente en no hacer nada
// Si se usa en memetico_generacional el resultado es un algoritmo genético generacional
fn nada<Trng: Rng>(_entrenamiento: &[Dato], _cromosomas: &mut ArbolBinario, _rng: &mut Trng) -> usize {
    return 0; // No hace nada. Por consiguiente, no debe incrementar el contador de evaluaciones
}

// Aplica búsqueda local a todos los cromosomas de una población
fn bl_todos<Trng: Rng>(entrenamiento: &[Dato], cromosomas: &mut ArbolBinario, rng: &mut Trng) -> usize {
    let max_evaluaciones = 2*entrenamiento[0].num_atributos();
    let num_cromosomas = cromosomas.len();
    let mut nuevos_cromosomas: ArbolBinario = cromosomas.iter().map(|(cr_id, cr)| {
        let nuevo_cromosoma = busqueda_local_generica_desde(&entrenamiento, &cr, &vecino_bl, max_evaluaciones, MAX_CICLOS_BL, rng);
        ((OrderedFloat(-evaluar(&entrenamiento, &nuevo_cromosoma)), cr_id.1), nuevo_cromosoma)
    }).collect();

    cromosomas.clear();
    cromosomas.append(&mut nuevos_cromosomas);

    num_cromosomas*(1 + max_evaluaciones)  // Se devuelve el número de evaluaciones, que está fijado para cada cromosoma en el guion
}

// Aplica búsqueda local a los cromosomas de una población con un 10% de probabilidad
fn bl_01<Trng: Rng>(entrenamiento: &[Dato], cromosomas: &mut ArbolBinario, rng: &mut Trng) -> usize {
    let max_evaluaciones = 2*entrenamiento[0].num_atributos();
    let mut cromosomas_borrados = vec![]; // Almacena las claves de los cromosomas que van a ser borrados
    let mut nuevos_cromosomas: ArbolBinario = cromosomas.iter().filter_map(|(cr_id, cr)| {
        if rng.gen::<f64>() >= 0.1 {
            None  // 90% de no devolver nada (y por tanto el cromosoma no se cambia)
        } else {
            // En el resto de casos, se cambia el cromosoma
            cromosomas_borrados.push(*cr_id);
            let nuevo_cromosoma = busqueda_local_generica_desde(&entrenamiento, &cr, &vecino_bl, max_evaluaciones, MAX_CICLOS_BL, rng);
            Some(((OrderedFloat(-evaluar(&entrenamiento, &nuevo_cromosoma)), cr_id.1), nuevo_cromosoma))
        }
    }).collect();

    let num_cromosomas = cromosomas_borrados.len();
    for cr_id in cromosomas_borrados {
        cromosomas.remove(&cr_id);  // Borramos los cromosomas que han sido explorados usando su clave
    }
    cromosomas.append(&mut nuevos_cromosomas);

    num_cromosomas*(1 + max_evaluaciones)
}

// Aplica búsqueda local al diez por ciento de los mejores cromosomas de una población
fn bl_01mej<Trng: Rng>(entrenamiento: &[Dato], cromosomas: &mut ArbolBinario, rng: &mut Trng) -> usize {
    let max_evaluaciones = 2*entrenamiento[0].num_atributos();
    let num_cromosomas = (cromosomas.len() as f64 * 0.1).round() as usize;
    let mut cromosomas_borrados = vec![]; // Almacena las claves de los cromosomas que van a ser borrados
    // Como el árbol de cromosomas está ordenado con los mejores primero, tomamos los primeros
    let mut nuevos_cromosomas: ArbolBinario = cromosomas.iter().take(num_cromosomas).map(|(cr_id, cr)| {
        cromosomas_borrados.push(*cr_id);
        let nuevo_cromosoma = busqueda_local_generica_desde(&entrenamiento, &cr, &vecino_bl, max_evaluaciones, MAX_CICLOS_BL, rng);
        ((OrderedFloat(-evaluar(&entrenamiento, &nuevo_cromosoma)), cr_id.1), nuevo_cromosoma)
    }).collect();

    for cr_id in cromosomas_borrados {
        cromosomas.remove(&cr_id);  // Borramos los cromosomas que han sido explorados usando su clave
    }
    cromosomas.append(&mut nuevos_cromosomas);

    num_cromosomas*(1 + max_evaluaciones)
}


// Ejecuta un algoritmo memético basado en un algoritmo genético
//    según un esquema generacional con elitismo
// Recibe la muestra, una función generadora de soluciones iniciales,
//   un operador de cruce, un operador de generación de un vecino para efectuar mutaciones,
//   un operador memético (que devuelve el número de evaluaciones usadas) y un RNG
// El procedimiento generador de soluciones iniciales debe no ser determinista
pub fn memetico_generacional<Trng: Rng>(entrenamiento: &[Dato], gen_iniciales: &Fn(&[Dato], &mut Trng) -> Vec<f64>, cruce: &Fn(&[f64], &[f64], &mut Trng) -> Vec<Vec<f64>>, vecino: &Fn(&[f64], usize, &mut Trng) -> Vec<f64>, op_memetica: (usize, &Fn(&[Dato], &mut ArbolBinario, &mut Trng) -> usize), rng: &mut Trng) -> Vec<f64> {
    // Rellenamos la población con elementos seleccionados por gen_iniciales (probablemente aleatorios)
    let mut poblacion = ArbolBinario::new();  // Por cada elemento, su evaluación y su identificador como clave (se ordenará según su evaluación, y en caso de empate según identificador)
    for i in 0..TAMANO_GENERACIONAL {
        let cromosoma_aleatorio = gen_iniciales(&entrenamiento, rng);
        poblacion.insert((OrderedFloat(-evaluar(&entrenamiento, &cromosoma_aleatorio)), i), cromosoma_aleatorio);
    }

    // Parámetros que afectan al operador de mutación
    let n_caracteristicas = entrenamiento[0].num_atributos();
    let n_genes = n_caracteristicas*TAMANO_GENERACIONAL;
    let esperanza_mutaciones = (TASA_MUTACION_GEN*(n_genes as f64)).round() as usize;

    let n_evaluaciones = RefCell::<usize>::new(TAMANO_GENERACIONAL); // número de veces que se ha evaluado la función objetivo
    let mut n_generaciones = 0;   // Número de generaciones desde la última vez que se aplicó el procedimiento memético

    loop {
        // El vector de nuevos cromosomas almacena también su identificador
        //   y su evaluación si no es un cromosoma nuevo
        let mut nueva_poblacion: Vec<(Option<(OrderedFloat<f64>, usize)>, Vec<f64>)> = Vec::with_capacity(TAMANO_GENERACIONAL);

        while nueva_poblacion.len() < TAMANO_GENERACIONAL {
            let mejor = |c1: (_, _), c2: (_, _)| -> (_, bool) {
                if c1.0 <= c2.0 { (c1, true) } else { (c2, false) }
            }; // Devuelve el mejor de dos cromosomas y true si es el primero. En caso de empate devuelve el más antiguo. Recuérdese que se almacena el opuesto de la evaluación del cromosoma

            // Permutamos los cromosomas (referencias a ellos, concretamente) para enfrentarlos y combinarlos
            let mut emparejamientos: Vec<_> = poblacion.iter().collect();
            rng.shuffle(&mut emparejamientos);

            // Hacemos grupos de 4. Se cruza el mejor de los dos primeros con el mejor de
            //   los dos segundos hasta que hagamos cruces para cubrir el 70% de la población
            let mut emp_iter = emparejamientos.into_iter();
            for _t in 0..((TAMANO_GENERACIONAL as f64)*TASA_CRUCE_GENERACIONAL/4.0).round() as usize {
                let grupo: Vec<(_, &Vec<f64>)> = emp_iter.by_ref().take(4).collect();
                let padre1 = mejor(grupo[0], grupo[1]).0;
                let padre2 = mejor(grupo[2], grupo[3]).0;
                let resultado_cruce = cruce(&padre1.1, &padre2.1, rng);
                let num_nuevos = resultado_cruce.len();
                for nuevos in resultado_cruce {
                    nueva_poblacion.push((None, nuevos));
                }
                if num_nuevos == 1 {
                    nueva_poblacion.push((Some(*padre2.0), padre2.1.to_vec())); // Si solo hay un cromosoma hijo, se añade uno de los padres
                }
            }
            // Del resto se selecciona el mejor de cada pareja sin hacer cruces
            for p in emp_iter.collect::<Vec<_>>().chunks(2) {
                if p.len() == 1 {
                    nueva_poblacion.push((Some(*p[0].0), p[0].1.to_vec()));
                } else {
                    let m = mejor(p[0], p[1]);
                    nueva_poblacion.push((Some(if m.1 { *p[0].0 } else { *p[1].0 } ), (m.0).1.to_vec()));
                }
            }
        };

        // Generamos un número aleatorio por cada mutación y mutamos con el operador de
        // búsqueda local definido para la práctica 1
        let numero_a_mutacion = |n: usize| {
            let n_mod = n % n_genes;
            (n_mod / n_caracteristicas, n_mod % n_caracteristicas)
        }; // Esta closure obtiene a partir de un número el individuo y el cromosoma mutados
        let mutaciones: Vec<_> = rng.gen_iter().take(esperanza_mutaciones).map(numero_a_mutacion).collect();
        for m in mutaciones {
            let (individuo, gen) = m;
            nueva_poblacion[individuo] = (None, vecino(&nueva_poblacion[individuo].1, gen, rng));
        }

        // Reemplazamos los cromosomas de la población por los nuevos
        // Si tenían tupla evaluación-identificador, se sigue usando. Si no, se evalúa y se asigna una
        let mut vieja_poblacion = ArbolBinario::new();
        std::mem::swap(&mut vieja_poblacion, &mut poblacion);

        for (_i, c) in nueva_poblacion.iter().enumerate() {
            poblacion.insert(c.0.unwrap_or_else(|| {
                  *n_evaluaciones.borrow_mut() += 1;
                  (OrderedFloat(-evaluar(&entrenamiento, &c.1)), *n_evaluaciones.borrow())
                }),
                (*c.1).to_vec());
        }

        // Elitismo: nos aseguramos de que el mejor cromosoma de la generación anterior se mantiene
        let mut elite_anteriores = vieja_poblacion.iter();
        let mejor_anterior = elite_anteriores.next().unwrap();

        if !poblacion.contains_key(&mejor_anterior.0) {    // Si la mejor solución anterior no estaba,
            poblacion.insert(*mejor_anterior.0, mejor_anterior.1.to_vec()); // la introducimos
        }
        if poblacion.len() == 1 + TAMANO_GENERACIONAL { // Si al introducir el mejor cromosoma anterior hay de más,
            let peor = *poblacion.iter().next_back().unwrap().0;  // eliminamos el peor
            poblacion.remove(&peor);
        }

        // Si se diese el caso de haber más de una repetición de cromosomas en la nueva generación,
        //   se sigue rellenando con los mejores cromosomas de la generación anterior
        while poblacion.len() < TAMANO_GENERACIONAL {
            let siguiente_mejor = elite_anteriores.next().unwrap();
            if !poblacion.contains_key(&siguiente_mejor.0) {
                poblacion.insert(*siguiente_mejor.0, siguiente_mejor.1.to_vec());
            }
        }

        // Aplicamos el procedimiento memético si corresponde, contando el número de evaluaciones adicionales
        n_generaciones += 1;
        if n_generaciones == op_memetica.0 {
            *n_evaluaciones.borrow_mut() += op_memetica.1(&entrenamiento, &mut poblacion, rng);
            n_generaciones = 0;
        }

        if *n_evaluaciones.borrow() >= MAX_EVALUACIONES {
            break;
        }
    };

    poblacion.iter().next().unwrap().1.to_vec() // Devolvemos la mejor solución encontrada
}


// Ejecuta un algoritmo genético según un esquema generacional con elitismo
// Recibe la muestra, una función generadora de soluciones iniciales, un operador de cruce,
//   un operador de generación de un vecino para efectuar mutaciones y un RNG
// El procedimiento generador de soluciones iniciales debe no ser determinista
pub fn genetico_generacional<Trng: Rng>(entrenamiento: &[Dato], gen_iniciales: &Fn(&[Dato], &mut Trng) -> Vec<f64>, cruce: &Fn(&[f64], &[f64], &mut Trng) -> Vec<Vec<f64>>, vecino: &Fn(&[f64], usize, &mut Trng) -> Vec<f64>, rng: &mut Trng) -> Vec<f64> {
    // Usa la función que aplica un algoritmo memético sin efectuar ningún procedimiento de explotación
    return memetico_generacional(&entrenamiento, &gen_iniciales, &cruce, &vecino, (99999999, &nada), rng);
}


// Ejecuta un algoritmo genético con esquema estacionario
// Recibe la muestra, una función generadora de soluciones iniciales, un operador de cruce,
//   un operador de generación de un vecino para efectuar mutaciones y un RNG
// El procedimiento generador de soluciones iniciales debe no ser determinista
pub fn genetico_estacionario<Trng: Rng>(entrenamiento: &[Dato], gen_iniciales: &Fn(&[Dato], &mut Trng) -> Vec<f64>, cruce: &Fn(&[f64], &[f64], &mut Trng) -> Vec<Vec<f64>>, vecino: &Fn(&[f64], usize, &mut Trng) -> Vec<f64>, rng: &mut Trng) -> Vec<f64> {
    // Rellenamos la población con elementos seleccionados por gen_iniciales (probablemente aleatorios)
    let mut poblacion: Vec<(Vec<f64>, f64)> = Vec::with_capacity(TAMANO_ESTACIONARIO);
    for _i in 0..TAMANO_ESTACIONARIO {
        let cromosoma = gen_iniciales(&entrenamiento, rng);
        poblacion.push((cromosoma.clone(), evaluar(&entrenamiento, &cromosoma)));
    }

    let mut peor; // (posición, evaluación) del peor cromosoma
    macro_rules! encontrar_peor {
        () => { peor = { let t = poblacion.iter().enumerate().min_by_key(|x| OrderedFloat((x.1).1)).unwrap(); (t.0, (t.1).1)} };
    }  // Esta macro fija la variable peor a una tupla con la posición y la valoración del peor cromosoma
    encontrar_peor!();

    let n_caracteristicas = entrenamiento[0].num_atributos();
    let n_evaluaciones = RefCell::<usize>::new(TAMANO_ESTACIONARIO); // número de veces que se ha evaluado la función objetivo

    loop {
        let mut tasa_mutacion = TASA_MUTACION_GEN;
        let hijos = {
            // Escogemos cuatro elementos al azar y los enfrentamos dos a dos
            // Puede resultar que se cruza un cromosoma consigo mismo. Si eso sucede, lo mutamos más de lo normal
            let i_1a = rng.gen_range(0, TAMANO_ESTACIONARIO);
            let i_1b = rng.gen_range(0, TAMANO_ESTACIONARIO);
            let i_2a = rng.gen_range(0, TAMANO_ESTACIONARIO);
            let i_2b = rng.gen_range(0, TAMANO_ESTACIONARIO);

            let padre1 = &poblacion[if poblacion[i_1a].1 >= poblacion[i_1b].1 { i_1a } else { i_1b }].0;
            let padre2 = &poblacion[if poblacion[i_2a].1 >= poblacion[i_2b].1 { i_2a } else { i_2b }].0;

            if padre1 == padre2 {       // Si los dos padres son el mismo,
                tasa_mutacion *= 100.0; // disparamos la tasa de mutaciones
            }
            // Cruzamos los dos ganadores e insertamos los hijos en la población si son mejores que el peor actual
            cruce(&padre1, &padre2, rng)
        };

        for h in hijos {
            // Decidimos las mutaciones de cada gen una por una y las efectuamos utilizando
            // como operador de mutación la búsqueda local definido para la práctica 1
            let mut h_mutado = h;
            for gen in 0..n_caracteristicas {
                if rng.gen::<f64>() < tasa_mutacion {
                    h_mutado = vecino(&h_mutado, gen, rng);  // Esto sobreescribe el vector de pesos, pero es infrecuente
                }
            }

            // Introducimos el hijo si no es peor que el actual peor, y eliminamos el peor
            let ev_h = evaluar(&entrenamiento, &h_mutado);
            *n_evaluaciones.borrow_mut() += 1;
            if ev_h > peor.1 {
                poblacion[peor.0] = (h_mutado, ev_h);
                encontrar_peor!();
            }
        }

        if *n_evaluaciones.borrow() >= MAX_EVALUACIONES {
            break;
        }
    };


    poblacion.iter().max_by_key(|x| OrderedFloat(x.1)).unwrap().0.clone()
}


// Algoritmo genético generacional con cruce BLX-0.3
pub fn agg_blx<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    genetico_generacional(&entrenamiento, &vector_au, &blx_03, &vecino_bl, rng)
}

// Algoritmo genético generacional con cruce aritmético
pub fn agg_ca<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    genetico_generacional(&entrenamiento, &vector_au, &ca, &vecino_bl, rng)
}

// Algoritmo genético estacionario con cruce BLX-0.3
pub fn age_blx<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    genetico_estacionario(&entrenamiento, &vector_au, &blx_03, &vecino_bl, rng)
}

// Algoritmo genético estacionario con cruce aritmético
pub fn age_ca<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    genetico_estacionario(&entrenamiento, &vector_au, &ca, &vecino_bl, rng)
}

// Algoritmo memético AM-(10, 1.0)
// Cada 10 generaciones aplica la búsqueda local de la práctica 1 a todos los elementos de la población
pub fn am_a<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    memetico_generacional(&entrenamiento, &vector_au, &blx_03, &vecino_bl, (10, &bl_todos), rng)
}

// Algoritmo memético AM-(10, 0.1)
// Cada 10 generaciones aplica la búsqueda local de la práctica 1 al 10% de los elementos de la población
pub fn am_b<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    memetico_generacional(&entrenamiento, &vector_au, &blx_03, &vecino_bl, (10, &bl_01), rng)
}

// Algoritmo memético AM-(10, 0.1mej)
// Cada 10 generaciones aplica la búsqueda local de la práctica 1 al 10% de los elementos de la población
pub fn am_c<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    memetico_generacional(&entrenamiento, &vector_au, &blx_03, &vecino_bl, (10, &bl_01mej), rng)
}



// Implementaciones de algoritmos adicionales


// Operador de cruce aritmético alternativo: si un gen es menor que 0.2 para
//   algún padre y la media es mayor o igual que 0.2, hay 10% de que se devuelva el mínimo
// Recibe los padres y un RNG y devuelve un hijo
fn ca_alt<Trng: Rng>(padre1: &[f64], padre2: &[f64], rng: &mut Trng) -> Vec<Vec<f64>> {
    let mut unico_hijo = padre1.iter().zip(padre2.iter()).map(|(x, y)| {
        let mut valor = (*x + *y)/2.0;
        let menor = if *x <= *y { *x } else { *y };
        if valor >= 0.2 && menor < 0.2 {
            if rng.gen::<f64>() < 0.1 { valor = menor }
        };
        valor
    }).collect();
    normalizar(&mut unico_hijo);  // Normalizamos el vector, porque seguramente el máximo ha dejado de ser 1
    vec![unico_hijo]
}


// Algoritmo genético estacionario con cruce aritmético alternativo
pub fn age_ca_alt<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    genetico_estacionario(&entrenamiento, &vector_au, &ca_alt, &vecino_bl, rng)
}


// Algoritmo genético generacional con cruce BLX-0.3 y operador de mutación alternativo
// El operador de mutación es el operador de vecino propuesto en la práctica 1
pub fn agg_blx_mut2<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    genetico_generacional(&entrenamiento, &vector_au, &blx_03, &vecino_bl_mut2, rng)
}


// Algoritmo genético estacionario con cruce BLX-0.3 y operador de mutación alternativo
// El operador de mutación es el operador de vecino propuesto en la práctica 1
pub fn age_blx_mut2<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    genetico_estacionario(&entrenamiento, &vector_au, &blx_03, &vecino_bl_mut2, rng)
}


// Aplica búsqueda local por afinidad a los cromosomas de una población con un 10% de probabilidad
// Véase el método potencia_optima en las funciones de la práctica 1
fn af_01<Trng: Rng>(entrenamiento: &[Dato], cromosomas: &mut ArbolBinario, rng: &mut Trng) -> usize {
    let mut num_evaluaciones = 0;
    let mut cromosomas_borrados = vec![]; // Almacena las claves de los cromosomas que van a ser borrados
    let mut nuevos_cromosomas: ArbolBinario = cromosomas.iter().filter_map(|(cr_id, cr)| {
        if rng.gen::<f64>() >= 0.1 {
            None  // 90% de no devolver nada (y por tanto el cromosoma no se cambia)
        } else {
            // En el resto de casos, se cambia el cromosoma
            cromosomas_borrados.push(*cr_id);
            let nuevo_cromosoma = afinidad_optima(&entrenamiento, &cr, rng);
            num_evaluaciones += 1 + cr.iter().filter(|w| **w != 0.0 && **w != 1.0).count();
            Some(((OrderedFloat(-evaluar(&entrenamiento, &nuevo_cromosoma)), cr_id.1), nuevo_cromosoma))
        }
    }).collect();

    for cr_id in cromosomas_borrados {
        cromosomas.remove(&cr_id);  // Borramos los cromosomas que han sido explorados usando su clave
    }
    cromosomas.append(&mut nuevos_cromosomas);

    num_evaluaciones
}

// Algoritmo memético AM-(10, 0.1,af)
// Cada 10 generaciones aplica la optimización por afinidad a los elementos de la población con probabilidad 10%
pub fn am_afinidad_01<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    memetico_generacional(&entrenamiento, &vector_au, &blx_03, &vecino_bl, (10, &af_01), rng)
}
