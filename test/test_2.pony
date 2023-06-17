# ../build/bin/pony ../test/test_2.pony -emit=token

def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {

  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  print(c);
  
}