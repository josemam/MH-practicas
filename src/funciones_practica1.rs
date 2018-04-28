use knn;    // Implementa el clasificador K-NN
use std;    // Usaremos su BTreeMap para ordenar los atributos en uno de los algoritmos

use knn::Dato;
use evaluacion_pesos::evaluar;
use ordered_float::OrderedFloat;
use rand::Rng;
use rand::distributions::{Sample, Normal};



// Algunas constantes y funciones auxiliares


// Normaliza un vector de pesos para que sus valores estén en [0, 1]
// Aplica una función lineal de forma que el máximo pasa a tomar el valor 1
pub fn normalizar(w: &mut Vec<f64>) {
    let max = w.iter().max_by_key(|x| OrderedFloat(**x)).unwrap().clone();
    if max != 1.0 && max != 0.0 {
        for wi in w.iter_mut() {
            *wi /= max;
        }
    }
}


// Genera un vector de pesos aleatorios de cierto tamaño
// Los pesos están en [0, 1] y siempre hay un peso de valor 1
// Se utiliza en los procedimientos de búsqueda local como solución inicial
fn vector_aleatorio_uniforme<Trng: Rng>(n_elementos: usize, rng: &mut Trng) -> Vec<f64> {
    let mut w: Vec<f64> = Vec::with_capacity(n_elementos); // pesos a devolver
    for _i in 0..n_elementos {
        w.push(rng.gen());  // Rellenamos la solución inicial con valores aleatorios en [0, 1]
    };
    normalizar(&mut w);     // Normalizamos de forma lineal: el peso máximo pasará a ser 1
    w
}


// Criterios de parada en búsqueda local: número de evaluaciones de la función objetivo y número
// de vecinos generados por cada componente a partir de una sola solución sin que haya mejora
// El procedimiento de búqueda local terminará cuando uno de estos dos criterios se cumpla
const MAX_EVALUACIONES: usize = 15000; // Tope de evaluaciones de la función objetivo antes de terminar
const MAX_CICLOS: usize = 20;          // Tope de veces que se explorará cada atributo sin que haya mejora



// Implementaciones de los algoritmos de aprendizaje de pesos
// Reciben un conjunto de entrenamiento y un generador de números aleatorios
// Devuelven un vector de pesos
// Pueden recibir cualquier tipo de generador de números aleatorios, aunque
//   en evaluacion_pesos.rs se usa un tipo de RNG particular para probarlos


// Combina un algoritmo de búsqueda de soluciones con otro que consista en una búsqueda local
// Concretamente, aplica la búsqueda local indicada al resultado del primero algoritmo
// Por ejemplo, se puede usar para aplicar una búsqueda local a partir de una solución aleatoria,
//   o a partir del resultado de RELIEF
pub fn combinar<Trng: Rng>(entrenamiento: &[Dato], algoritmo_1: &Fn(&[Dato], &mut Trng) -> Vec<f64>, algoritmo_bl: &Fn(&[Dato], &[f64], &mut Trng) -> Vec<f64>, rng: &mut Trng) -> Vec<f64> {
    algoritmo_bl(&entrenamiento, &algoritmo_1(&entrenamiento, rng), rng)
}


// Devuelve como solución un vector aleatorio uniforme. Se usará como solución inicial para la BL
// Está estructurado como un algoritmo por cuestiones de legibilidad de código
pub fn vector_au<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    vector_aleatorio_uniforme(entrenamiento[0].num_atributos(), rng)
}



// Ejecuta una búsqueda local de soluciones a partir de una dada con criterios de parada y
//   un procedimiento de generación de soluciones vecinas indicados a través de parámetros
// El orden de los atributos es el mismo en el que vienen en los datos
pub fn busqueda_local_generica_desde<Trng: Rng>(entrenamiento: &[Dato], w_base: &[f64], vecino: &Fn(&[f64], usize, &mut Trng) -> Vec<f64>, max_evaluaciones: usize, max_ciclos: usize, rng: &mut Trng) -> Vec<f64> {
    // Generamos una solución inicial
    let n_atributos = entrenamiento[0].num_atributos();
    let mut w = w_base.to_vec();
    let mut fw = evaluar(&entrenamiento, &w); // Puntuación de la mejor solución

    let mut atr = 0;      // posición del próximo atributo que va a ser mutado
    let mut n_ciclos = 0; // número mínimo de veces que se ha probado cada atributo desde la última mejora

    for _i in 1..max_evaluaciones {   // Ya se ha hecho una evaluación de la función objetivo, aquí se hace una menos
        // Generamos un vecino con el operador recibido como parámetro
        let nw = vecino(&w, atr, rng);
        // Comprobamos si hemos obtenido una solución mejor
        let fnw = evaluar(&entrenamiento, &nw);
        if fnw > fw { // El vecino es mejor que el anterior
            w = nw;
            fw = fnw;
            atr = 0;
            n_ciclos = 0;
        } else {
            atr += 1;
            if atr == n_atributos {
                atr = 0;
                n_ciclos += 1;
                if n_ciclos == max_ciclos {
                    break;
                }
            }
        }
    };

    w
}


// Operador de vecino descrito en el guion
pub fn vecino_bl<Trng: Rng>(w: &[f64], i: usize, rng: &mut Trng) -> Vec<f64> {
    let mut nw = w.to_vec();
    let valor_previo = nw[i];
    nw[i] += Normal::new(0.0, 0.3).sample(rng);

    if nw[i] < 0.0 {
        nw[i] = 0.0;
    } else if nw[i] > 1.0 {
        nw[i] = 1.0;
    }
    if valor_previo == 1.0 && nw[i] != 1.0 {
        normalizar(&mut nw);
    }
    nw
}


// Ejecuta búsqueda local de soluciones a partir de una dada con el procedimiento descrito en el guion
// El orden en el que se mutan los atributos es el mismo en el que vienen en los datos
pub fn busqueda_local_desde<Trng: Rng>(entrenamiento: &[Dato], w_base: &[f64], rng: &mut Trng) -> Vec<f64> {
    busqueda_local_generica_desde(&entrenamiento, &w_base, &vecino_bl, MAX_EVALUACIONES, MAX_CICLOS, rng)
}

// Ejecuta búsqueda local con el procedimiento descrito en el guion
// Parte de un vector aleatorio
pub fn busqueda_local<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    combinar(&entrenamiento, &vector_au, &busqueda_local_desde, rng)
}


// Ejecuta el algoritmo greedy RELIEF para obtener un vector de pesos
// Requiere que todos los atributos sean valores reales
pub fn relief<Trng: Rng>(entrenamiento: &[Dato], _rng: &mut Trng) -> Vec<f64> {
    let n_atributos = entrenamiento[0].num_atributos();
    let mut w = vec![0.0; n_atributos]; // pesos a devolver, inicialmente a 0
    let w_euc = vec![1.0; n_atributos]; // pesos uniformes, para calcular las instancias más cercanas con la distancia euclídea

    // Modificamos los pesos con el procedimiento del amigo y enemigo más cercano
    for ei in entrenamiento {
        let ee = knn::get_enemigo_mas_cercano(&entrenamiento, &ei, &w_euc);
        let ea = knn::  get_amigo_mas_cercano(&entrenamiento, &ei, &w_euc);
        for i in 0..w.len() {
            w[i] += (ei[i] - ee[i]).abs() - (ei[i] - ea[i]).abs();  // Esto no funciona si hay atributos categóricos
        }
    }

    // Devolvemos el resultado normalizado y truncando los valores negativos
    let wmax = w.iter().cloned().fold(0./0., f64::max); // Esto devuelve el máximo; es feo porque Rust tiene cuidado con los flotantes
    w.iter().map(|p| if *p <= 0.0 { 0.0 } else { p/wmax }).collect()
}


// Devuelve un vector de pesos todos a 1
// Está estructurado como un algoritmo por cuestiones de legibilidad de código
pub fn uno_nn<Trng: Rng>(entrenamiento: &[Dato], _rng: &mut Trng) -> Vec<f64> {
    vec![1.0; entrenamiento[0].num_atributos()]
}



// Implementaciones de algoritmos adicionales

// Calcula el truncado óptimo para unos pesos en un conjunto de entrenamiento
// Se trata de encontrar un valor tal que, anulando todos los pesos menores o
//   iguales que dicho valor, se obtenga la máxima puntuación
// Para ello se prueba a truncar en todos los valores distintos de 0 y 1
pub fn truncado_optimo<Trng: Rng>(entrenamiento: &[Dato], w_base: &[f64], _rng: &mut Trng) -> Vec<f64> {
    fn truncar(pesos: &[f64], corte: f64) -> Vec<f64> {
        pesos.iter().map(|p| if *p <= corte { 0.0 } else { *p } ).collect()
    } // Esta es la función que trunca un vector de pesos

    let mut mejor_cut = 0.19999999; // Valor con el que se obtiene el mejor corte. Truncando con este valor inicial no se afecta a la clasificación
    let mut mejor_pts = evaluar(&entrenamiento, &w_base);

    for w in w_base {
        if *w >= 0.2 && *w != 1.0 {
            let candidato_cut = *w;  // Fijamos el corte al valor de w: así no cuenta el peso w ni ninguno menor
            let candidato_pts = evaluar(&entrenamiento, &truncar(w_base, candidato_cut));

            if candidato_pts > mejor_pts {
                mejor_cut = candidato_cut;
                mejor_pts = candidato_pts;
            }
        }
    }

    truncar(&w_base, mejor_cut) // Devolvemos los pesos truncados con el mejor valor de corte que se ha encontrado
}

// Calcula el exponente óptimo para unos pesos en un conjunto de entrenamiento
// Se trata de encontrar un valor tal que, elevando todos los pesos a dicho valor,
//   se obtenga la máxima puntuación (en particular, habrá más o menos pesos menores que 0.2)
// Para ello se prueba a elevar el vector a los exponentes con los que cada uno de
//   los valores (distintos de 0 y 1) pasa a tomar un valor ligeramente inferior a 0.2
pub fn potencia_optima<Trng: Rng>(entrenamiento: &[Dato], w_base: &[f64], _rng: &mut Trng) -> Vec<f64> {
    fn elevar(pesos: &[f64], exp: f64) -> Vec<f64> {
        pesos.iter().map(|p| (*p).powf(exp)).collect()
    } // Esta es la función que eleva un vector de pesos componente a componente

    let mut mejor_exp = 1.0; // Exponente con el que se obtiene la mejor clasificación. Con 1.0 no se cambia nada
    let mut mejor_pts = evaluar(&entrenamiento, &w_base);

    for w in w_base {
        if *w != 0.0 && *w != 1.0 {
            // Fijamos el exponente al número al que hay que elevar w para obtener
            //   poco menos que 0.2: así no cuenta el peso w ni ninguno menor
            let candidato_exp = (0.1999999f64).log(*w);
            let candidato_pts = evaluar(&entrenamiento, &elevar(w_base, candidato_exp));

            if candidato_pts > mejor_pts {
                mejor_exp = candidato_exp;
                mejor_pts = candidato_pts;
            }
        }
    }

    elevar(&w_base, mejor_exp) // Devolvemos los pesos elevados al mejor exponente que se ha encontrado
}

// Calcula la afinidad óptima para unos pesos en un conjunto de entrenamiento
// Se trata de encontrar un valor tal que, transformando de forma afín ese valor en 0.2 y el 1 en
//   el 1, se obtenga la máxima puntuación (en particular, habrá más o menos pesos menores que 0.2)
// Para ello se prueba a transformar el vector a los exponentes con los que cada uno de
//   los valores (distintos de 0 y 1) pasa a tomar un valor ligeramente inferior a 0.2
pub fn afinidad_optima<Trng: Rng>(entrenamiento: &[Dato], w_base: &[f64], _rng: &mut Trng) -> Vec<f64> {
    fn truncar_01(f: f64) -> f64 { if f >= 0.0 { if f < 1.0 { f } else { 1.0 } } else { 0.0 } }
    fn transformar(pesos: &[f64], corte: f64) -> Vec<f64> {
        pesos.iter().map(|p| truncar_01((0.8*(*p) - corte + 0.2)/(1.0 - corte))).collect()
    } // Esta es la función que transforma un vector de pesos componente a componente

    let mut mejor_cut = 0.2; // Valor que va a 0.2 con el que se obtiene la mejor clasificación. Con 0.2 no se cambia nada
    let mut mejor_pts = evaluar(&entrenamiento, &w_base);

    for w in w_base {
        if *w != 0.0 && *w < 0.9999999 {
            let candidato_cut = *w + 0.0000001f64;  // Fijamos el valor que va a 0.2 a poco más del valor de w: así no cuenta el peso w ni ninguno menor
            let candidato_pts = evaluar(&entrenamiento, &transformar(w_base, candidato_cut));

            if candidato_pts > mejor_pts {
                mejor_cut = candidato_cut;
                mejor_pts = candidato_pts;
            }
        }
    }

    transformar(&w_base, mejor_cut) // Devolvemos los pesos tras aplicarles la mejor transformación que se ha encontrado
}


// Ejecuta RELIEF y aplica al resultado el truncamiento óptimo
// Al igual que RELIEF, requiere que todos los atributos sean valores reales
pub fn relief_truncado<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    combinar(&entrenamiento, &relief, &truncado_optimo, rng)
}


// Ejecuta RELIEF y aplica al resultado el exponente óptimo
// Al igual que RELIEF, requiere que todos los atributos sean valores reales
pub fn relief_potencia<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    combinar(&entrenamiento, &relief, &potencia_optima, rng)
}


// Ejecuta RELIEF y aplica al resultado la transformación afín óptima
// Al igual que RELIEF, requiere que todos los atributos sean valores reales
pub fn relief_afinidad<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    combinar(&entrenamiento, &relief, &afinidad_optima, rng)
}


// Operador de vecino alternativo al propuesto en el guion
// En lugar de truncar el valor de la componente modificada si toma un valor mayor que 1,
//   se normaliza el vector. Esto puede hacer que varios elementos bajen rápidamente del
//   umbral 0.2, pudiendo obtener soluciones más simples rápidamente.
pub fn vecino_bl_mut2<Trng: Rng>(w: &[f64], i: usize, rng: &mut Trng) -> Vec<f64> {
    let mut nw = w.to_vec();
    let valor_previo = nw[i];
    nw[i] += Normal::new(0.0, 0.3).sample(rng);

    if nw[i] < 0.0 {
        nw[i] = 0.0;
    }
    if valor_previo == 1.0 || nw[i] > 1.0 {
        normalizar(&mut nw);
    }
    nw
}


// Ejecuta búsqueda local de soluciones a partir de una dada con un procedimiento de mutación distinto
// El orden de los atributos es el mismo en el que vienen en los datos
pub fn busqueda_local_mut2_desde<Trng: Rng>(entrenamiento: &[Dato], w_base: &[f64], rng: &mut Trng) -> Vec<f64> {
    busqueda_local_generica_desde(&entrenamiento, &w_base, &vecino_bl_mut2, MAX_EVALUACIONES, MAX_CICLOS, rng)
}

// Ejecuta búsqueda local con un procedimiento de mutación distinto
// Parte de un vector aleatorio
pub fn busqueda_local_mut2<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    combinar(&entrenamiento, &vector_au, &busqueda_local_mut2_desde, rng)
}



// Ejecuta búsqueda local de soluciones a partir de una dada con criterios de parada y 
//   un procedimiento de generación de soluciones vecinas indicados a través de parámetros
// Los atributos que por sí solos clasifican mejor la muestra de entrenamiento se exploran primero
pub fn busqueda_local_ordenada_desde<Trng: Rng>(entrenamiento: &[Dato], w_base: &[f64], vecino: &Fn(&[f64], usize, &mut Trng) -> Vec<f64>, max_evaluaciones: usize, max_ciclos: usize, rng: &mut Trng) -> Vec<f64> {
    // Generamos una solución inicial
    let n_atributos = entrenamiento[0].num_atributos();
    let mut w = w_base.to_vec();
    let mut fw = evaluar(&entrenamiento, &w); // Puntuación de la mejor solución

    let mut atr_id = 0;   // posición (en el vector ordenado) del próximo atributo que va a ser mutado
    let mut n_ciclos = 0; // número mínimo de veces que se ha probado cada atributo desde la última mejora

    // Ordenamos los atributos por la tasa de clasificación de los datos usando solo ellos mismos
    let peso_i = |i| -> Vec<f64> {
        let mut pesos = vec![0.0; n_atributos];
        pesos[i] = 1.0;
        pesos
    };  // Devuelve un vector con el peso i-ésimo a 1, el resto a 0
    // Almacenamos en un árbol de búsqueda binaria la posición de cada componente y dicha tasa
    let mut arbol_atributos = std::collections::BTreeMap::<_, Vec<usize>>::new();
    for a in 0..n_atributos {
        arbol_atributos.entry(OrderedFloat(-evaluar(&entrenamiento, &peso_i(a))))
                       .or_insert_with(Vec::new).push(a); // Si hay un atributo con la misma valoración, se añade a su vector. Si no, se crea uno
    }
    let mut indices_atributos = Vec::new();
    for (_, va) in &arbol_atributos {
        for a in va {
            indices_atributos.push(*a);
        }
    }

    for _i in 1..max_evaluaciones {   // Ya se ha hecho una evaluación de la función objetivo, aquí se hace una menos
        let atr = indices_atributos[atr_id];
        // Generamos un vecino con el operador recibido como parámetro
        let nw = vecino(&w, atr, rng);
        // Comprobamos si hemos obtenido una solución mejor
        let fnw = evaluar(&entrenamiento, &nw);
        if fnw > fw { // El vecino es mejor que el anterior
            w = nw;
            fw = fnw;
            atr_id = 0;
            n_ciclos = 0;
        } else {
            atr_id += 1;
            if atr_id == n_atributos {
                atr_id = 0;
                n_ciclos += 1;
                if n_ciclos == max_ciclos {
                    break;
                }
            }
        }
    };

    w
}


// Ejecuta búsqueda local de soluciones a partir de una dada con un criterio de ordenación de atributos
// Los atributos que por sí solos clasifican mejor la muestra de entrenamiento se exploran primero
pub fn busqueda_local_orden_desde<Trng: Rng>(entrenamiento: &[Dato], w_base: &[f64], rng: &mut Trng) -> Vec<f64> {
    busqueda_local_ordenada_desde(&entrenamiento, &w_base, &vecino_bl, MAX_EVALUACIONES, MAX_CICLOS, rng)
}

// Ejecuta búsqueda local con un criterio de ordenación de atributos
// Parte de un vector aleatorio
pub fn busqueda_local_orden<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    combinar(&entrenamiento, &vector_au, &busqueda_local_orden_desde, rng)
}


// Ejecuta búsqueda local de soluciones a partir de una dada con el criterio de
//   ordenación de atributos y con el operador de mutación alternativo
pub fn busqueda_local_orden_mut2_desde<Trng: Rng>(entrenamiento: &[Dato], w_base: &[f64], rng: &mut Trng) -> Vec<f64> {
    busqueda_local_ordenada_desde(&entrenamiento, &w_base, &vecino_bl_mut2, MAX_EVALUACIONES, MAX_CICLOS, rng)
}

// Ejecuta búsqueda local con el criterio de ordenación de atributos y con
//   el operador de mutación alternativo
// Parte de un vector aleatorio
pub fn busqueda_local_orden_mut2<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    combinar(&entrenamiento, &vector_au, &busqueda_local_orden_mut2_desde, rng)
}
