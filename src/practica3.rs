#[macro_use] extern crate clap;       // Gestor de argumentos del programa
#[macro_use] extern crate itertools;  // Iteradores avanzados
extern crate rand;          // Generador de números aleatorios
extern crate ordered_float; // Implementación de orden total en flotantes (tiene en cuenta la existencia de NaN)
extern crate byteorder;     // Permite interpretar arrays de 8 bits como de 64 bits independientemente de la máquina

mod knn;                    // Implementa el clasificador K-NN
mod evaluacion_pesos;       // Contiene las funciones para evaluar los distintos algoritmos
mod funciones_practica1;
mod funciones_practica3;

use byteorder::{ByteOrder, BigEndian};
use funciones_practica3::*;   // Usamos todas las funciones implementadas para la práctica 3, lógicamente
use funciones_practica1::*;



// Prueba un conjunto de datos con los algoritmos implementados e imprime los resultados
fn test(archivo: &str, semilla: &[u64]) {
    // Abrimos el archivo manejando posibles errores
    let datos = knn::leer_archivo(archivo).unwrap_or_else(|e| {
          println!("No se pudo abrir el archivo {}: {}", archivo, e); Vec::new()
        });
    if datos.is_empty() { return }

    let lista_algoritmos: Vec<(fn(&[knn::Dato], &mut _) -> Vec<f64>, &str)> = vec![
            (uno_nn, "1NN"),
            (relief, "RELIEF"),
            (es, "Enfriamiento simulado"),
            (ils, "ILS"),
            (de_rand_1, "DE/rand/1"),
            (de_ctb_1, "DE/current-to-best/1"),
            (relief_truncado, "RELIEF + truncado"),
            (relief_potencia, "RELIEF + potencia"),
            (relief_afinidad, "RELIEF + afinidad"),
            (es_mut2, "ES_mut2"),
            (es_prop, "ES_prop"),
            (es_prop_mut2, "ES_prop_mut2"),
            (ils_mut2, "ILS_mut2"),
            (ils_afinidad, "ILS_afinidad"),
            (ils_afinidad_relief, "ILS_afinidad_relief")
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
        (about: "Implementación de la práctica 3\n\nPrueba los algoritmos pedidos con el archivo de datos indicado, utilizando una semilla que se obtiene a partir de una cadena de texto.\n\nLa semilla se inicializa al mismo valor antes de ejecutar cada uno de los algoritmos.")
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
