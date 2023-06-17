# ../build/bin/pony ../test/test_10.pony -emit=ast
# ../build/bin/pony ../test/test_10.pony -emit=jit

def main() {

  var a = [[1, 2, 3], [4, 5, 6]];
  var b<2, 3> = [1, 2, 3, 4, 5, 6];
  var d[2][3] = [1, 2, 3, 4, 5, 6];
  print(b);
  
}