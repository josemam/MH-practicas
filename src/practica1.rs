#[macro_use]                // Permite el uso de una macro en clap
extern crate clap;          // Gestor de argumentos del programa
extern crate rand;          // Generador de números aleatorios
extern crate ordered_float; // Implementación de orden total en flotantes (tiene en cuenta la existencia de NaN)
extern crate byteorder;     // Permite interpretar arrays de 8 bits como de 64 bits independientemente de la máquina

mod knn;                    // Implementa el clasificador K-NN
mod evaluacion_pesos;       // Contiene las funciones para evaluar los distintos algoritmos

use knn::Dato;
use evaluacion_pesos::evaluar;
use ordered_float::OrderedFloat;
use rand::Rng;
use rand::distributions::{Sample, Normal};
use byteorder::{ByteOrder, BigEndian};



// Algunas constantes y funciones auxiliares


// Normaliza un vector de pesos para que sus valores estén en [0, 1]
// Aplica una función lineal de forma que el máximo pasa a tomar el valor 1
fn normalizar(w: &mut Vec<f64>) {
    let max = w.iter().max_by_key(|x| OrderedFloat(**x)).unwrap().clone();
    for wi in w.iter_mut() {
        *wi /= max;
    };
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


// Ejecuta búsqueda local de soluciones con el procedimiento descrito en el guion
// El orden en el que se mutan los atributos es el mismo en el que vienen en los datos
pub fn busqueda_local<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    // Generamos una solución inicial
    let n_atributos = entrenamiento[0].num_atributos();
    let mut distribucion = Normal::new(0.0, 0.3); // Función de distribución para las mutaciones de una componente
    let mut w = vector_aleatorio_uniforme(n_atributos, rng);
    let mut fw = evaluar(&entrenamiento, &w); // Puntuación de la mejor solución

    let mut atr = 0;      // posición del próximo atributo que va a ser mutado
    let mut n_ciclos = 0; // número mínimo de veces que se ha probado cada atributo desde la última mejora

    for _i in 1..MAX_EVALUACIONES {   // Ya se ha hecho una evaluación de la función objetivo, aquí se hace una menos
        // Generamos un vecino
        let atr_previo = w[atr];
        w[atr] += distribucion.sample(rng);  // Mutamos la componente atr
        
        if w[atr] < 0.0 {
            w[atr] = 0.0;
        } else if w[atr] > 1.0 {
            w[atr] = 1.0;
        }

        // Comprobamos si hemos obtenido una solución mejor
        let fw_nuevo = evaluar(&entrenamiento, &w);
        if fw_nuevo > fw { // El vecino es mejor que el anterior
            fw = fw_nuevo;
            atr = 0;
            n_ciclos = 0;
        } else {
            w[atr] = atr_previo;
            atr += 1;
            if atr == n_atributos {
                atr = 0;
                n_ciclos += 1;
                if n_ciclos == MAX_CICLOS {
                    break;
                }
            }
        }
    };

    w
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


// Combina un algoritmo de búsqueda de soluciones con un criterio que modifica dichas soluciones
// Por ejemplo, se puede usar para aplicar un truncamiento mayor que 0.2 a las soluciones de RELIEF
pub fn combinar<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng, algoritmo: &Fn(&[Dato], &mut Trng) -> Vec<f64>, criterio: &Fn(&[Dato], &[f64]) -> Vec<f64>) -> Vec<f64> {
    criterio(&entrenamiento, &algoritmo(&entrenamiento, rng))
}

// Calcula el truncado óptimo para unos pesos en un conjunto de entrenamiento
// Se trata de encontrar un valor tal que, anulando todos los pesos menores o
//   iguales que dicho valor, se obtenga la máxima puntuación
// Para ello se prueba a truncar en todos los valores distintos de 0 y 1
pub fn truncado_optimo(entrenamiento: &[Dato], w_base: &[f64]) -> Vec<f64> {
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
pub fn potencia_optima(entrenamiento: &[Dato], w_base: &[f64]) -> Vec<f64> {
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


// Ejecuta RELIEF y aplica al resultado el truncamiento óptimo
// Al igual que RELIEF, requiere que todos los atributos sean valores reales
pub fn relief_truncado<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    combinar(&entrenamiento, rng, &relief, &truncado_optimo)
}


// Ejecuta RELIEF y aplica al resultado el exponente óptimo
// Al igual que RELIEF, requiere que todos los atributos sean valores reales
pub fn relief_potencia<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    combinar(&entrenamiento, rng, &relief, &potencia_optima)
}


// Ejecuta búsqueda local de soluciones con un procedimiento de mutación distinto
// En lugar de truncar el valor de la componente modificada si toma un valor mayor que 1,
//   se normaliza el vector. Esto puede hacer que varios elementos bajen rápidamente del
//   umbral 0.2, pudiendo obtener soluciones más simples rápidamente.
// El orden de los atributos es el mismo en el que vienen en los datos
pub fn busqueda_local_mut2<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    // Generamos una solución inicial
    let n_atributos = entrenamiento[0].num_atributos();
    let mut distribucion = Normal::new(0.0, 0.3); // Función de distribución para las mutaciones de una componente
    let mut w = vector_aleatorio_uniforme(n_atributos, rng);
    let mut fw = evaluar(&entrenamiento, &w); // Puntuación de la mejor solución

    let mut atr = 0;      // posición del próximo atributo que va a ser mutado
    let mut n_ciclos = 0; // número mínimo de veces que se ha probado cada atributo desde la última mejora

    for _i in 1..MAX_EVALUACIONES {   // Ya se ha hecho una evaluación de la función objetivo, aquí se hace una menos
        // Generamos un vecino
        // La clonación solo es necesaria si se va a normalizar, pero como es rápida comparada con la evaluación, se ha optado por efectuarla siempre
        let mut nw = w.clone();
        let nw_atr_previo = nw[atr];
        nw[atr] += distribucion.sample(rng);  // Mutamos la componente atr
        
        if nw[atr] < 0.0 {
            nw[atr] = 0.0;
        }
        if nw_atr_previo == 1.0 || nw[atr] > 1.0 { // Nótese que esta condición puede darse a la vez que nw[atr] < 0.0
            normalizar(&mut nw);
        }

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
                if n_ciclos == MAX_CICLOS {
                    break;
                }
            }
        }
    };

    w
}


// Ejecuta búsqueda local de soluciones con un criterio de ordenación de atributos
// Los atributos que por sí solos clasifican mejor la muestra de entrenamiento se exploran primero
pub fn busqueda_local_orden<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    // Generamos una solución inicial
    let n_atributos = entrenamiento[0].num_atributos();
    let mut distribucion = Normal::new(0.0, 0.3); // Función de distribución para las mutaciones de una componente
    let mut w = vector_aleatorio_uniforme(n_atributos, rng);
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

    for _i in 1..MAX_EVALUACIONES {   // Ya se ha hecho una evaluación de la función objetivo, aquí se hace una menos
        let atr = indices_atributos[atr_id];
        // Generamos un vecino
        let atr_previo = w[atr];
        w[atr] += distribucion.sample(rng);

        if w[atr] < 0.0 {
            w[atr] = 0.0;
        } else if w[atr] > 1.0 {
            w[atr] = 1.0;
        }

        // Comprobamos si hemos obtenido una solución mejor
        let fw_nuevo = evaluar(&entrenamiento, &w);
        if fw_nuevo > fw { // El vecino es mejor que el anterior
            fw = fw_nuevo;
            atr_id = 0;
            n_ciclos = 0;
        } else {
            w[atr] = atr_previo;
            atr_id += 1;
            if atr_id == n_atributos {
                atr_id = 0;
                n_ciclos += 1;
                if n_ciclos == MAX_CICLOS {
                    break;
                }
            }
        }
    };

    w
}


// Ejecuta búsqueda local de soluciones con el criterio de ordenación de atributos y con
//   el operador de mutación alternativo
pub fn busqueda_local_orden_mut2<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    // Generamos una solución inicial
    let n_atributos = entrenamiento[0].num_atributos();
    let mut distribucion = Normal::new(0.0, 0.3); // Función de distribución para las mutaciones de una componente
    let mut w = vector_aleatorio_uniforme(n_atributos, rng);
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

    for _i in 1..MAX_EVALUACIONES {   // Ya se ha hecho una evaluación de la función objetivo, aquí se hace una menos
        let atr = indices_atributos[atr_id];
        // Generamos un vecino
        // La clonación solo es necesaria si se va a normalizar, pero como es rápida comparada con la evaluación, se ha optado por efectuarla siempre
        let mut nw = w.clone();
        let nw_atr_previo = nw[atr];
        nw[atr] += distribucion.sample(rng);  // Mutamos la componente atr
        
        if nw[atr] < 0.0 {
            nw[atr] = 0.0;
        }
        if nw_atr_previo == 1.0 || nw[atr] > 1.0 { // Nótese que esta condición puede darse a la vez que nw[atr] < 0.0
            normalizar(&mut nw);
        }

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
                if n_ciclos == MAX_CICLOS {
                    break;
                }
            }
        }
    };

    w
}



// Prueba un conjunto de datos con los algoritmos implementados e imprime los resultados
fn test(archivo: &str, semilla: &[u64]) {
    // Abrimos el archivo manejando posibles errores
    let datos = knn::leer_archivo(archivo).unwrap_or_else(|e| {
          println!("No se pudo abrir el archivo {}: {}", archivo, e); Vec::new()
        });
    if datos.is_empty() { return }

    let lista_algoritmos: Vec<(fn(&[Dato], &mut _) -> Vec<f64>, &str)> = vec![
            (uno_nn, "1NN"),
            (relief, "RELIEF"),
            (busqueda_local, "búsqueda local"),
            (relief_truncado, "RELIEF + truncado"),
            (relief_potencia, "RELIEF + potencia"),
            (busqueda_local_mut2, "BL con otra mutación"),
            (busqueda_local_orden, "BL con orden"),
            (busqueda_local_orden_mut2, "BL con orden y otra mutación")
        ];
    for algoritmo in &lista_algoritmos {
        println!("\n{} sobre los datos en {}...", algoritmo.1, archivo);
        evaluacion_pesos::ffcv(&algoritmo.0, &datos, semilla);
    }
}



fn main() {
    // Gestor de argumentos
    let matches = clap_app!(practica1 =>
        (author: crate_authors!())
        (about: "Implementación de la práctica 1\n\nPrueba los algoritmos pedidos con el archivo de datos indicado, utilizando una semilla que se obtiene a partir de una cadena de texto.\n\nLa semilla se inicializa al mismo valor antes de ejecutar cada uno de los algoritmos.")
        (set_term_width: 79)
        (@arg INPUT: "Archivo .arff con los datos de entrada. Si no se indica se efectúa con los tres archivos indicados en la práctica")
        (@arg semilla: -s --seed +takes_value "Texto del que obtener la semilla para el PRNG. Puede necesitar comillas si contiene espacios")
    ).get_matches();

    // Leemos una semilla como texto y la transformamos a slice de
    // enteros de 64 bits pasando por slice de enteros de 8 bits
    let semilla_texto = if matches.is_present("semilla") {
            matches.value_of("semilla").unwrap()
        } else {
            // Semilla por defecto, en caso de que el usuario decida no introducir ninguna
            "Es el usuario el que elige a la semilla y es la semilla la que quiere que sean los usuarios la semilla."
        };
    let mut semilla_bytes = semilla_texto.to_string().into_bytes();
    for _i in 0..(((-(semilla_bytes.len() as isize))%8)&7) {
        semilla_bytes.push(0);  // Nos aseguramos de que el número de bytes es múltiplo de 8 rellenando con ceros
    }
    let mut semilla_vec64 = vec![0; semilla_bytes.len() / 8];
    BigEndian::read_u64_into(&semilla_bytes[..], &mut semilla_vec64);
    let semilla = semilla_vec64.as_slice();

    if matches.is_present("INPUT") {
        test(matches.value_of("INPUT").unwrap(), &semilla); // Ejecutamos todos los algoritmos con el archivo indicado, si lo hay
    } else {
        for archivo in &["instances/ozone-320.arff", "instances/parkinsons.arff", "instances/spectf-heart.arff"] {
            test(archivo, &semilla);  // Si no se indica archivo, se usan los tres ofrecidos en la práctica
        }
    }
}
