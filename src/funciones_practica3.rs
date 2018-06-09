use knn::Dato;
use evaluacion_pesos::evaluar;
use rand::Rng;
use rand::distributions::{Sample, Normal};
use std::mem::swap;

use funciones_practica1::*;    // Recuperamos la búsqueda local



// Algunas constantes y funciones auxiliares


const MAX_EVALUACIONES: usize = 15000;    // Máximo de evaluaciones de la función objetivo en cada algoritmo
const MAX_EVALUACIONES_BL: usize = 1000;  // Cada búsqueda local en ILS efectúa este número de evaluaciones
const MAX_CICLOS_BL: usize = 100000000;   // No hay límite de ciclos, sino de evaluaciones
const FACTOR_VECINOS_ES: usize = 10;      // Factor del número de vecinos a explorar por iteración de ES
const TAMANO_DE: usize = 50;              // Tamaño de la población en evolución diferencial
const DE_CR: f64 = 0.5;                   // Tasa de cruce por gen de evolución diferencial
const DE_F : f64 = 0.5;                   // Factor de evolución diferencial


type PoblacionDE = (Vec<(f64, Vec<f64>)>, usize); // Tipo de dato de una población de DE. Se almacena el índice del mejor cromosoma

// Aplica enfriamiento según un esquema de enfriamiento de Cauchy modificado
fn enfriamiento_cauchy(actual: f64, t_inicial: f64, t_final: f64, num_iteraciones: usize) -> f64 {
    let b = (t_inicial - t_final)/((num_iteraciones as f64)*t_inicial*t_final);
    actual/(1.0 + b*actual)
}


// Procedimiento de mutación para ILS
// Una décima parte de las componentes son mutadas según una normal de desviación típica 0.4
fn vecino_ils<Trng: Rng>(w: &[f64], rng: &mut Trng) -> Vec<f64> {
    let mut nw = w.to_vec();

    let mut componentes: Vec<usize> = (0..w.len()).collect();
    let mut sl = &mut componentes;
    rng.shuffle(&mut sl);
    for i in sl.iter().take((w.len() as f64 / 10.0).round() as usize) {
        nw[*i] += Normal::new(0.0, 0.4).sample(rng);

        if nw[*i] < 0.0 {
            nw[*i] = 0.0;
        } else if nw[*i] > 1.0 {
            nw[*i] = 1.0;
        }
    }

    // Normalizamos y devolvemos el vector mutado
    normalizar(&mut nw);
    nw
}


// Procedimiento de búsqueda local para ILS
// Es la misma búsqueda local de la práctica 1 salvo el criterio de parada: se hacen siempre 1000 evaluaciones
pub fn bl_ils<Trng: Rng>(entrenamiento: &[Dato], w_base: &[f64], rng: &mut Trng) -> Vec<f64> {
    busqueda_local_generica_desde(&entrenamiento, &w_base, &vecino_bl, MAX_EVALUACIONES_BL, MAX_CICLOS_BL, rng)
}





// Algoritmo de enfriamiento simulado general
// Recibe el procedimiento con el que se elige una solución inicial,
//   el operador de vecino y el esquema de enfriamiento
pub fn simulated_annealing_general<Trng: Rng>(entrenamiento: &[Dato], gen_inicial: &Fn(&[Dato], &mut Trng) -> Vec<f64>, vecino: &Fn(&[f64], usize, &mut Trng) -> Vec<f64>, enfriamiento: &Fn(f64, f64, f64, usize) -> f64, rng: &mut Trng) -> Vec<f64> {
    let n_caracteristicas = entrenamiento[0].num_atributos();
    let max_vecinos = FACTOR_VECINOS_ES*n_caracteristicas;        // Máximo de vecinos en cada iteración
    let max_exitos = (0.1*max_vecinos as f64).ceil() as usize;    // Máximo de éxitos en cada iteración
    let num_iteraciones = ((MAX_EVALUACIONES-1) as f64/max_vecinos as f64).ceil() as usize; // Número de iteraciones

    let solucion_aleatoria = gen_inicial(&entrenamiento, rng);
    let mut solucion_actual = (solucion_aleatoria.clone(), evaluar(&entrenamiento, &solucion_aleatoria));
    let mut mejor_solucion = solucion_actual.clone();

    let t_inicial: f64 = - 0.3 * solucion_actual.1 / (0.3_f64).ln();
    let t_final  : f64 = 0.001;
    let mut temperatura = t_inicial;

    for _i in 0..num_iteraciones {
        let mut exitos_restantes = max_exitos;
        for _j in 0..max_vecinos {
            if exitos_restantes == 0 { break; }
            let c = rng.gen_range(0, n_caracteristicas);  // Escogemos una característica
            let nueva_solucion = vecino(&solucion_actual.0, c, rng);
            let ev = evaluar(&entrenamiento, &nueva_solucion);
            let diferencia = solucion_actual.1 - ev;  // Si es negativa, la nueva solución es mejor (mayor evaluación)
            if diferencia < 0.0 || rng.gen::<f64>() <= (-diferencia / temperatura).exp() {  // K = 1.0
                exitos_restantes -= 1;
                solucion_actual = (nueva_solucion.clone(), ev);
                if ev > mejor_solucion.1 {
                    mejor_solucion = solucion_actual.clone();
                }
            }
        }

        if exitos_restantes == max_exitos { break; }  // Si no hay mejora en una iteración, terminamos

        temperatura = enfriamiento(temperatura, t_inicial, t_final, num_iteraciones);
    }

    mejor_solucion.0
}

// Algoritmo de enfriamiento simulado partiendo de una solución aleatoria
// Utiliza el operador de vecino de la práctica 1 como operador de mutación
//   y un esquema de enfriamiento de Cauchy modificado
pub fn es<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    simulated_annealing_general(&entrenamiento, &vector_au, &vecino_bl, &enfriamiento_cauchy, rng)
}


// Algoritmo de búsqueda local iterativa general
// Recibe el procedimiento con el que se elige una solución inicial,
//   el operador de mutación y el procedimiento de búsqueda
pub fn iterated_local_search_general<Trng: Rng>(entrenamiento: &[Dato], gen_inicial: &Fn(&[Dato], &mut Trng) -> Vec<f64>, mutacion_brusca: &Fn(&[f64], &mut Trng) -> Vec<f64>, bl: &Fn(&[Dato], &[f64], &mut Trng) -> Vec<f64>, rng: &mut Trng) -> Vec<f64> {
    let solucion = gen_inicial(&entrenamiento, rng);
    let solucion_bl = bl(&entrenamiento, &solucion, rng);
    let mut mejor_solucion = (solucion_bl.clone(), evaluar(&entrenamiento, &solucion_bl));
    for _i in 1..(MAX_EVALUACIONES / MAX_EVALUACIONES_BL) {
        let solucion_mutada = mutacion_brusca(&mejor_solucion.0, rng);
        let nueva_solucion = bl(&entrenamiento, &solucion_mutada, rng);
        let evaluacion = evaluar(&entrenamiento, &nueva_solucion);
        if evaluacion > mejor_solucion.1 {
            mejor_solucion = (nueva_solucion.clone(), evaluacion);
        }
    } // Se hacen 1001 evaluaciones en cada bucle: se evalúa una vez más para puntuar el resultado de la BL

    mejor_solucion.0
}

// Algoritmo de búsqueda local reiterada con la búsqueda local de la práctica 1
//   y el operador de mutación brusco descrito en este guion
pub fn ils<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    iterated_local_search_general(&entrenamiento, &vector_au, &vecino_ils, &bl_ils, rng)
}

// Algoritmo de evolución diferencial general
// Recibe el procedimiento con el que se genera cada elemento de
//   la población inicial y el operador de evolución diferencial concreto
pub fn differential_evolution_general<Trng: Rng>(entrenamiento: &[Dato], gen_inicial: &Fn(&[Dato], &mut Trng) -> Vec<f64>, operador_de: &Fn(&PoblacionDE, usize, &mut Trng) -> (Option<f64>, Vec<f64>), rng: &mut Trng) -> Vec<f64> {
    // Inicializamos y evaluamos la población, guardando el índice del mejor
    let mut poblacion: PoblacionDE = (vec![], 0);
    for i in 0..TAMANO_DE {
        let nuevo_cromosoma = gen_inicial(&entrenamiento, rng);
        let ev = evaluar(&entrenamiento, &nuevo_cromosoma);
        poblacion.0.push((ev, nuevo_cromosoma));
        if ev > poblacion.0[poblacion.1].0 {
            poblacion.1 = i;  // Si el nuevo es mejor que el anterior mejor, pasa a ser el mejor
        }
    }
    let mut evaluaciones_restantes = MAX_EVALUACIONES - TAMANO_DE;

    while evaluaciones_restantes > 0 {
        let mut nueva_poblacion = poblacion.clone();
        for i in 0..TAMANO_DE {
            if evaluaciones_restantes == 0 { break; }
            let nuevo_c = operador_de(&poblacion, i, rng);
            if nuevo_c.0.is_none() {
                let ev = evaluar(&entrenamiento, &nuevo_c.1);
                evaluaciones_restantes -= 1;
                if ev > poblacion.0[i].0 {
                    nueva_poblacion.0[i] = (ev, nuevo_c.1);
                    if ev > nueva_poblacion.0[nueva_poblacion.1].0 {
                        nueva_poblacion.1 = i;  // Si el nuevo es el mejor, se marca como tal
                    }
                }
            }
        }
        swap(&mut poblacion, &mut nueva_poblacion);
    }

    // Devolvemos el mejor cromosoma encontrado
    poblacion.0[poblacion.1].1.to_vec()
}

// Obtiene un número entero aleatorio distinto de ciertos números
// Uso: aleatorio_distinto!(rng, máximo; número a descartar 1, número a descartar 2, ...)
macro_rules! aleatorio_distinto {
    ( $rng:ident, $m:expr; $( $n:expr ),* ) => {
        {
            let mut r: usize = 0;
            let mut repetido = true;
            while repetido {
                repetido = false;
                r = $rng.gen_range(0, $m);
                $(
                    if r == $n {
                        repetido = true;
                    }
                )*
            };
            r
        }
    }
}

// Operación de mutación del algoritmo DE/rand/1
pub fn op_rand_1<Trng: Rng>(poblacion: &PoblacionDE, actual: usize, rng: &mut Trng) -> (Option<f64>, Vec<f64>) {
    let tam_poblacion = poblacion.0.len();
    let num_cromosomas = poblacion.0[0].1.len();

    let p1 = aleatorio_distinto!(rng, tam_poblacion; actual);
    let p2 = aleatorio_distinto!(rng, tam_poblacion; actual, p1);
    let p3 = aleatorio_distinto!(rng, tam_poblacion; actual, p1, p2);

    let mut nuevo_cromosoma = (Some(poblacion.0[actual].0), poblacion.0[actual].1.to_vec());
    for n in 0..num_cromosomas {
        if rng.gen::<f64>() < DE_CR {
            nuevo_cromosoma.0 = None;
            nuevo_cromosoma.1[n] = poblacion.0[p1].1[n] + DE_F * (poblacion.0[p2].1[n] - poblacion.0[p3].1[n]);
            if nuevo_cromosoma.1[n] < 0.0 {
                nuevo_cromosoma.1[n] = 0.0;
            } else if nuevo_cromosoma.1[n] > 1.0 {
                nuevo_cromosoma.1[n] = 1.0;
            }
        }
    }

    nuevo_cromosoma
}

// Operación de mutación del algoritmo DE/current-to-best/1
pub fn op_ctb_1<Trng: Rng>(poblacion: &PoblacionDE, actual: usize, rng: &mut Trng) -> (Option<f64>, Vec<f64>) {
    let tam_poblacion = poblacion.0.len();
    let num_cromosomas = poblacion.0[0].1.len();

    let p1 = aleatorio_distinto!(rng, tam_poblacion; actual, poblacion.1);
    let p2 = aleatorio_distinto!(rng, tam_poblacion; actual, poblacion.1, p1);

    let mut nuevo_cromosoma = (Some(poblacion.0[actual].0), poblacion.0[actual].1.to_vec());
    for n in 0..num_cromosomas {
        if rng.gen::<f64>() < DE_CR {
            nuevo_cromosoma.0 = None;
            nuevo_cromosoma.1[n] += DE_F * (poblacion.0[poblacion.1].1[n] - nuevo_cromosoma.1[n] + poblacion.0[p1].1[n] - poblacion.0[p2].1[n]);
            if nuevo_cromosoma.1[n] < 0.0 {
                nuevo_cromosoma.1[n] = 0.0;
            } else if nuevo_cromosoma.1[n] > 1.0 {
                nuevo_cromosoma.1[n] = 1.0;
            }
        }
    }

    nuevo_cromosoma
}

// Algoritmo de evolución diferencial con el operador DE/rand/1
pub fn de_rand_1<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    differential_evolution_general(&entrenamiento, &vector_au, &op_rand_1, rng)
}

// Algoritmo de evolución diferencial con el operador DE/current-to-best/1
pub fn de_ctb_1<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    differential_evolution_general(&entrenamiento, &vector_au, &op_ctb_1, rng)
}



// Implementaciones de algoritmos adicionales


// Aplica enfriamiento según un esquema proporcional
// Después de aplicarse num_iteraciones veces a partir de la temperatura inicial se tiene la final
fn enfriamiento_proporcional(actual: f64, t_inicial: f64, t_final: f64, num_iteraciones: usize) -> f64 {
    let b = (t_final/t_inicial).powf(1.0 / num_iteraciones as f64); // La idea es que t_inicial * b^num_iteraciones == t_final
    actual*b
}

// Procedimiento de búsqueda local para ILS con el operador de mutación alternativo propuesto en la práctica 1
pub fn bl_ils_mut2<Trng: Rng>(entrenamiento: &[Dato], w_base: &[f64], rng: &mut Trng) -> Vec<f64> {
    busqueda_local_generica_desde(&entrenamiento, &w_base, &vecino_bl_mut2, MAX_EVALUACIONES_BL, MAX_CICLOS_BL, rng)
}

// Algoritmo de enfriamiento simulado partiendo de una solución aleatoria
// Utiliza el operador de vecino alternativo propuesto en la práctica 1 como operador de mutación
//   y un esquema de enfriamiento de Cauchy modificado
pub fn es_mut2<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    simulated_annealing_general(&entrenamiento, &vector_au, &vecino_bl_mut2, &enfriamiento_cauchy, rng)
}

// Algoritmo de enfriamiento simulado partiendo de una solución aleatoria
// Utiliza el operador de vecino de la práctica 1 como operador de mutación
//   y un esquema de enfriamiento proporcional, de convergencia más lenta
pub fn es_prop<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    simulated_annealing_general(&entrenamiento, &vector_au, &vecino_bl, &enfriamiento_proporcional, rng)
}

// Algoritmo de enfriamiento simulado partiendo de una solución aleatoria
// Utiliza el operador de vecino alternativo propuesto en la práctica 1 como operador de mutación
//   y un esquema de enfriamiento proporcional, de convergencia más lenta
pub fn es_prop_mut2<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    simulated_annealing_general(&entrenamiento, &vector_au, &vecino_bl_mut2, &enfriamiento_proporcional, rng)
}

// Algoritmo de búsqueda local reiterada con la búsqueda local alternativa propuesta en la práctica 1
//   y el operador de mutación brusco descrito en este guion
pub fn ils_mut2<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    iterated_local_search_general(&entrenamiento, &vector_au, &vecino_ils, &bl_ils_mut2, rng)
}

// Algoritmo de búsqueda local reiterada con la búsqueda local por afinidad propuesta en la práctica 2
//   y el operador de mutación brusco descrito en este guion
pub fn ils_afinidad<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    iterated_local_search_general(&entrenamiento, &vector_au, &vecino_ils, &afinidad_optima, rng)
}

// Algoritmo de búsqueda local reiterada con la búsqueda local por afinidad propuesta en la práctica 2,
//   el operador de mutación brusco descrito en este guion y RELIEF como algoritmo de solución inicial
pub fn ils_afinidad_relief<Trng: Rng>(entrenamiento: &[Dato], rng: &mut Trng) -> Vec<f64> {
    iterated_local_search_general(&entrenamiento, &relief, &vecino_ils, &afinidad_optima, rng)
}
