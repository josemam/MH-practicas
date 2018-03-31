extern crate itertools;
extern crate ordered_float;
mod arff;

use self::arff::ArffContent;
use std::{path, io};
use std::collections::HashSet;
use ordered_float::OrderedFloat;
use self::itertools::multizip;    // Permite combinar las componentes de tres vectores
use std::f64::INFINITY;
use std::ops::{Index, IndexMut};
use std::ptr; // Comparación de punteros
use std::hash::{Hash, Hasher};


// Distancia entre variables categóricas: distancia de Hamming
fn distancia_cuadrado_c(a: &str, b: &str) -> f64 {
    if a == b { 0.0 } else { 1.0 }
}

// Distancia entre variables reales: diferencia real
fn distancia_cuadrado_f(a: f64, b: f64) -> f64 {
    let d = b - a;
    d*d
}

// Distancia entre vectores de características: distancia euclídea
fn distancia_cuadrado_vc(a: &[String], b: &[String], w: &[f64]) -> f64 {
    let mut d = 0.0;
    for (x, y, p) in multizip((a, b, w)) {
        if *p >= 0.2 {
            d += p*distancia_cuadrado_c(x, y);
        }
    }
    d
}

// Distancia entre vectores de reales: distancia euclídea
fn distancia_cuadrado_vf(a: &[f64], b: &[f64], w: &[f64]) -> f64 {
    let mut d = 0.0;
    for (x, y, p) in multizip((a, b, w)) {
        if *p >= 0.2 {
            d += p*distancia_cuadrado_f(*x, *y);
        }
    }
    d
}


// Tipo de dato para cada objeto de una muestra
#[derive(Clone)]
pub struct Dato {
    atributos_f: Vec<f64>,    // valores en los atributos reales
    atributos_c: Vec<String>, // valores en los atributos categóricos
    id_categoria: i32,        // id de la categoría con la que se corresponde el dato
}

impl Dato {
    pub fn new(af: &[f64], ac: &[String], cat: i32) -> Dato {
        Dato {
            atributos_f: af.to_vec(),
            atributos_c: ac.to_vec(),
            id_categoria: cat,
        }
    }

    pub fn num_atributos(&self) -> usize {
        self.atributos_f.len() + self.atributos_c.len()
    }

    pub fn id_categoria(&self) -> i32 {
        self.id_categoria
    }
}


impl Index<usize> for Dato {
    type Output = f64;

    fn index<'a>(&'a self, index: usize) -> &'a f64 {
        &self.atributos_f[index as usize]
    }
}

impl IndexMut<usize> for Dato {
    fn index_mut<'a>(&'a mut self, index: usize) -> &'a mut f64 {
        &mut self.atributos_f[index as usize]
    }
}

// Implementamos Eq y Hash para Dato para poder comprobar repetidos
impl PartialEq for Dato {
    fn eq(&self, otro: &Dato) -> bool {
        self.id_categoria == otro.id_categoria
          && self.atributos_f.len() == otro.atributos_f.len()
          && self.atributos_c.len() == otro.atributos_c.len()
          && self.atributos_f.iter().zip(otro.atributos_f.iter())
                .all(|(x, y)| (x == y) || (x.is_nan() && y.is_nan()))
          && self.atributos_c.iter().zip(otro.atributos_c.iter())
                .all(|(x, y)| x == y)
    }
}

impl Eq for Dato {}

impl Hash for Dato {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for f in self.atributos_f.iter() {
            OrderedFloat(*f).hash(state); // Aprovechamos que OrderedFloat sí tiene hash
        }
        self.atributos_f.len().hash(state);
        self.atributos_c.hash(state);
        self.id_categoria.hash(state);
    }
}


// Distancia (al cuadrado) entre las características de dos datos
fn distancia_cuadrado(a: &Dato, b: &Dato, w: &[f64]) -> f64 {
    let num_flotantes = a.atributos_f.len();
      (distancia_cuadrado_vf(&a.atributos_f, &b.atributos_f, &w[0..num_flotantes])
     + distancia_cuadrado_vc(&a.atributos_c, &b.atributos_c, &w[num_flotantes..]))
}

// Obtiene la categoría del dato más cercano a uno dado
// No comprueba que el dato a clasificar no esté en la muestra de entrenamiento:
//   si lo está se dará el propio dato
pub fn get_mas_cercano(vm: &[Dato], d: &Dato, w: &[f64]) -> i32 {
    vm.iter().min_by_key(|x| OrderedFloat(distancia_cuadrado(x, d, w))).unwrap().id_categoria
}

// Obtiene la categoría del dato más cercano a uno dado que no es él mismo
pub fn get_mas_cercano_distinto(vm: &[Dato], d: &Dato, w: &[f64]) -> i32 {
    vm.iter().min_by_key(
        |x| OrderedFloat(if ptr::eq(*x, d) { INFINITY } else { distancia_cuadrado(x, d, w) })
      ).unwrap().id_categoria
}

// Obtiene el dato más cercano a uno dado de la misma categoría que no es él mismo
// Asume que hay algún elemento de distinta categoría. Lanza excepción en caso contrario
pub fn get_amigo_mas_cercano<'a>(vm: &'a [Dato], d: &Dato, w: &[f64]) -> &'a Dato {
    let cat = d.id_categoria;
    vm.iter().min_by_key(
        |x| OrderedFloat(if x.id_categoria != cat || ptr::eq(*x, d) { INFINITY } else { distancia_cuadrado(x, d, w) })
      ).unwrap_or_else(|| panic!("No se encontraron elementos de la misma categoría que cierto dato"))
}

// Obtiene el dato más cercano a uno dado de distinta categoría
// Asume que hay algún elemento de distinta categoría. Lanza excepción en caso contrario
pub fn get_enemigo_mas_cercano<'a>(vm: &'a [Dato], d: &Dato, w: &[f64]) -> &'a Dato {
    let cat = d.id_categoria;
    vm.iter().min_by_key(
        |x| OrderedFloat(if x.id_categoria == cat { INFINITY } else { distancia_cuadrado(x, d, w) })
      ).unwrap_or_else(|| panic!("No se encontraron elementos de una categoría distinta a la de cierto dato"))
}


// Normaliza un vector de datos para que los valores de las características numéricas estén en [0, 1]
fn normalizar(v: &mut Vec<Dato>) {
    for c in 0..v[0].atributos_f.len() {
        let min = v.iter().min_by_key(|x| OrderedFloat(x[c])).unwrap()[c];
        let max = v.iter().max_by_key(|x| OrderedFloat(x[c])).unwrap()[c];
        let dif = max-min;
        for vi in v.iter_mut() {
            vi[c] = (vi[c]-min)/dif;
        };
    }
}

impl ArffContent {
    // Obtiene los datos a partir del formato que se obtiene en arff.rs
    // Asume que no hay variables categóricas que no sean la variable a estimar
    fn get_datos(&self) -> Vec<Dato> {
        let mut vm = vec![];
        let mut vm_set = HashSet::new();  // Controlaremos los elementos repetidos con un conjunto
        for d in &self.data {
            let af: Vec<f64> = (&d.values[..d.values.len()-1]).iter().map(|x| x.num().unwrap()).collect();
            let ac = vec![];
            let cat = d.values.last().unwrap().text().unwrap() as i32; // el formato en arff.rs asigna un identificador que empieza en 0
            let nuevo_dato = Dato::new(&af, &ac, cat);
            if !vm_set.contains(&nuevo_dato) {  // Agregamos el dato si no había sido agregado ya
                vm.push(nuevo_dato.clone());
                vm_set.insert(nuevo_dato);
            }
        }

        normalizar(&mut vm);
        vm
    }
}

// Obtiene los datos de un archivo .arff
pub fn leer_archivo(ruta: &str) -> Result<Vec<Dato>, io::Error> {
    let contenido = arff::ArffContent::new(path::Path::new(ruta));
    match contenido {
        Ok(c)  => Ok(c.get_datos()),
        Err(e) => Err(e), // Si new devolvió un error, devolvemos el mismo error
    }
}
