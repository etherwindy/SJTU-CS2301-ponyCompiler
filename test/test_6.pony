# ../build/bin/pony ../test/test_6.pony -emit=token


def main() {

   var a[2][3] = [1, 2.3., 3, 4, 5, 6];
   
}



def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {

  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  var d = multiply_transpose(transpose(a), c);
  print(d);
  
}