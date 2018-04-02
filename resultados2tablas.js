// Uso: node resultados2tablas.js

function procesa_linea(l) {
   return [l.split(":")[1].split("%")[0], l.split(",")[1].split("%")[0], l.split("Agregado:")[1].split(". ")[0], l.split("Tiempo:")[1].split(" ms")[0]].map(function(x) {return "$" + x + "$"})
}

function get_tablas(f) {
   var data = require('fs').readFileSync(f, 'utf8').split("\r\n").join("\n");
   var tests = ("\n" + data + "\n").split("\n\n").slice(1, -1);
   var conjuntos = {};
   var algoritmos = {};
   var splitter = " sobre los datos en "
   for (var t in tests) {
      var test = tests[t]
      var set = test.split(splitter)[1].split("...\n")[0]
      var alg = test.split(splitter)[0]
      if (!conjuntos.hasOwnProperty(set))
         conjuntos[set] = []
      if (!algoritmos.hasOwnProperty(alg))
         algoritmos[alg] = []
      var resultados = test.split("...\n")[1].split("\n")
      conjuntos[set].push([alg, resultados])
      algoritmos[alg].push([set, resultados])
   }

   var n_casos = Object.keys(conjuntos).length
   var tabla_global =  "\\begin{tabular}{c" + ("|r|r|r|r|").repeat(n_casos) + "}\n\\cline{2-" + (1+4*n_casos) + "}\n & \\multicolumn{4}{ |c|| }{" + Object.keys(conjuntos).join("} & \\multicolumn{4}{ |c|| }{").split("").reverse().join("").replace(" ||c| ", " |c|").split("").reverse().join("") + "}\\\\ \\cline{2-" + (1+4*n_casos) + "} \n" + ("& \\texttt{\\%\\_clas} & \\texttt{\\%\\_red} & Agr. & T (ms)").repeat(n_casos)

   var header_global = tabla_global
   
   for (var a in algoritmos) {
      var n_casos = algoritmos[a].length
      var tabla = header_global;
      for (var i = 0; i < algoritmos[a][0][1].length-1; i++) {
         tabla +=  "\\\\ \\hline \n\\multicolumn{1}{ |c|  }{Partición " + (1+i) + "}";
         for (var j = 0; j < algoritmos[a].length; j++)
           tabla += " &" + procesa_linea(algoritmos[a][j][1][i]).join(" & ")
      }
      tabla += "\\\\ \\hline\n\\hline\\multicolumn{1}{ |c|  }{\\large{Media}}";
      tabla_global += "\\\\ \\hline \n\\multicolumn{1}{ |c|  }{" + a + "}"
      for (var j = 0; j < algoritmos[a].length; j++) {
         var linea = "&" + procesa_linea(algoritmos[a][j][1][algoritmos[a][j][1].length-1]).join(" & ")
         tabla += linea
         tabla_global += linea
      }
      tabla += "\\\\ \\hline \n\\end{tabular}"
      require('fs').writeFile("latex/" + a + ".txt", tabla, function(err) {if (err) console.log("Error al escribir la tabla de " + a)})
   }

   tabla_global += "\\\\ \\hline \n\\end{tabular}"
   require('fs').writeFile("latex/tabla-global.txt", tabla_global, function(err) {if (err) console.log("Error al escribir la tabla global")})
}


get_tablas(process.argv.concat().slice(2).join(" "))
