extern crate time;
extern crate std;

use knn;
use knn::Dato;

use rand::{Rng, SeedableRng, Isaac64Rng}; // generadores de números aleatorios
use self::time::PreciseTime;    // medición de tiempo con resolución de 1 ns (aunque solo lo usaremos con precisión de 1 ms)


// Obtiene la distribución de las clases de una muestra como el total de elementos de cada clase
pub fn get_distribucion_clases(vm: &[Dato]) -> Vec<u32> {
    let mut contador: Vec<u32> = vec![];
    for m in vm {
        let cat = m.id_categoria();
        while cat >= contador.len() as i32 {
            contador.push(0);
        }
        contador[cat as usize] += 1;
    }
    contador
}

// Obtiene un vector con referencias a los datos de una clase a partir de su identificador
pub fn get_datos_de_clase<'a>(vm: &[&'a Dato], id: i32) -> Vec<&'a Dato> {
    vm.iter().filter(|m| m.id_categoria() == id).map(|m| *m).collect()
}


// Puntúa una distribución de pesos según su tasa de aciertos clasificando una muestra
//   con el criterio 1-NN (con leave-one-out si el conjunto de test es el de entrenamiento)
// Se devuelve el porcentaje de clases correctamente asignadas
// Asume que los pesos están normalizados: el máximo es 1
fn evaluar_clasificacion(entrenamiento: &[Dato], test: &[Dato], w: &[f64]) -> f64 {
    let instancias_test = test.len();
    let mut bien_clasificadas: i32 = 0;

    // Escoge el clasificador con leave-one-out si el conjunto de entrenamiento y el de prueba coinciden
    let clasificador =
        if (entrenamiento as *const _, entrenamiento.len()) == (test as *const _, test.len()) {
            knn::get_mas_cercano_distinto
        } else {
            knn::get_mas_cercano
        };

    // Clasifica los datos del conjunto de prueba y comprueba si se coincide con su clase correcta
    for dato in test {
        if clasificador(&entrenamiento, &dato, &w) == dato.id_categoria() {
            bien_clasificadas += 1;
        }
    }

    100.0 * (bien_clasificadas as f64) / (instancias_test as f64)
}

// Puntúa una distribución de pesos según su simplicidad
// Asume que los pesos están normalizados: el máximo es 1
fn evaluar_simplicidad(w: &[f64]) -> f64 {
    100.0 * (w.iter().filter(|p| **p < 0.2).count() as f64) / (w.len() as f64)
}

// Agrega la puntuación de una distribución según efectividad y simplicidad
fn evaluar_agregado(tasa_clas: f64, tasa_red: f64) -> f64 {
    let alpha: f64 = 0.5;
    alpha*tasa_clas + (1.0 - alpha)*tasa_red
}

// Puntúa una distribución de pesos según su tasa de aciertos en leave-one-out
//   clasificando una muestra y según su simplicidad
// Esta será la función objetivo usada por todos los algoritmos salvo el RELIEF
pub fn evaluar(datos: &[Dato], w: &[f64]) -> f64 {
    evaluar_agregado(evaluar_clasificacion(&datos, &datos, &w), evaluar_simplicidad(&w))
}

// Obtiene un vector de (cualquier tipo) a partir de un slice de referencias a (cualquier tipo)
macro_rules! desreferenciar {
    ( $x:expr ) => {
        {
            let v: Vec<_> = $x.iter().map(|m| (**m).clone()).collect();
            v
        }
    };
}

// Implementación de 5-fold cross validation
// Recibe una función que implemente un algoritmo que obtenga pesos de una muestra de entrenamiento,
//   los datos de entrenamiento y validación y una semilla para el PRNG
// Muestra por pantalla los parámetros pedidos: Tasa_clas, Tasa_red, Agregado y Tiempo,
//   tanto para cada uno de los tests como la media de estos en los cinco tests
// Se asume que el algoritmo devuelve los pesos debidamente normalizados
pub fn ffcv(algoritmo: &Fn(&[Dato], &mut Isaac64Rng) -> Vec<f64>, datos: &[Dato], seed: &[u64]){
    // Inicializamos un PRNG usando la semilla recibida
    // Escogemos Isaac64Rng, que implementa el algoritmo ISAAC-64, de Robert Jenkins
    // Véase https://docs.rs/rand/0.4.2/rand/struct.Isaac64Rng.html para una descripción del RNG
    //  y http://www.burtleburtle.net/bob/rand/isaacafa.html para información sobre el algoritmo
    let mut rng = Isaac64Rng::from_seed(seed);

    // Obtenemos una permutación de los datos que solo depende de la semilla utilizada
    // Nótese que el estado del PRNG se verá alterado tras esta operación a un valor
    //   que solo depende del estado anterior y del número de datos
    let mut datos_vr: Vec<&Dato> = datos.iter().collect();
    rng.shuffle(&mut datos_vr);

    // Dividimos los datos en cinco particiones con aproximadamente la misma distribución de clases
    let mut distribucion = get_distribucion_clases(datos);
    let n_fold = 5;   // Por si se quiere cambiar el número de particiones
    let mut particion: Vec<Vec<&Dato>> = vec![Vec::new(); n_fold];
    for cl_id in 0..distribucion.len() {
        // Seleccionamos los datos de cada clase y los repartimos en la partición
        let mut datos_clase = get_datos_de_clase(&datos_vr, cl_id as i32).into_iter();
        for i in 0..n_fold {
            let tomados = (distribucion[cl_id] as f64 /((n_fold-i) as f64)).round() as u32;
            let nuevos: Vec<&Dato> = datos_clase.by_ref().take(tomados as usize).collect();
            particion[i].extend(nuevos);
            distribucion[cl_id] -= tomados;
        }
    }

    let mut estadisticos: Vec<(f64, f64, f64, i64)> = Vec::new();
    let mut medias: Vec<f64> = vec![0.0; 4];

    for i in 0..n_fold {
        let test = desreferenciar!(&particion[i]);
        let mut entrenamiento: Vec<Dato> = Vec::new();
        for j in 0..n_fold { if j != i {
            entrenamiento.extend(desreferenciar!(&particion[j]));
        } };

        let t1 = PreciseTime::now();   // Tomamos el instante de tiempo inicial
        let pesos = algoritmo(&entrenamiento, &mut rng); // Ejecutamos el algoritmo y obtenemos los pesos
        let t2 = PreciseTime::now();   // Ídem con el final

        let tiempo_ms = t1.to(t2).num_milliseconds();

        let tasa_clas = evaluar_clasificacion(&entrenamiento, &test, &pesos); // Evaluamos los pesos en el conjunto de prueba
        let tasa_red = evaluar_simplicidad(&pesos); // Computamos la simplicidad de los pesos obtenidos
        let agregado = evaluar_agregado(tasa_clas, tasa_red);

        println!("Test {}: {:6.2}% aciertos, {:6.2}% reducción. Agregado: {:6.2}. Tiempo:{:6} ms",
                      1+i, tasa_clas, tasa_red, agregado, tiempo_ms);
        estadisticos.push((tasa_clas, tasa_red, agregado, tiempo_ms));
        for m in 0..4 {
            medias[m] += [tasa_clas, tasa_red, agregado, tiempo_ms as f64][m];
        }
    }

    for m in 0..4 {
        medias[m] /= n_fold as f64;
    }

    println!("Media : {:6.2}% aciertos, {:6.2}% reducción. Agregado: {:6.2}. Tiempo:{:6} ms", medias[0], medias[1], medias[2], medias[3].round());
}
