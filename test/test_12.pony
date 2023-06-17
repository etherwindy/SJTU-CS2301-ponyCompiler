# ../build/bin/pony ../test/test_12.pony -emit=ast
# ../build/bin/pony ../test/test_12.pony -emit=jit
def multiply_transpose(a, b) {
  return transpose(a) * transpose(b);
}

def main() {
  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var c = multiply_transpose(a, b);
  var d = multiply_transpose(a, b);
  var e = transpose(a)*c+transpose(b)+d;
  print(e);
  
}