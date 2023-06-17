//===- Parser.h - Pony Language Parser -------------------------------------===//
//
//===----------------------------------------------------------------------===//
//
// This file implements the parser for the Pony language. It processes the Token
// provided by the Lexer and returns an AST.
//
//===----------------------------------------------------------------------===//

#ifndef PONY_PARSER_H
#define PONY_PARSER_H

#include "pony/AST.h"
#include "pony/Lexer.h"

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <map>
#include <math.h>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace pony {

/// This is a simple recursive parser for the Pony language. It produces a well
/// formed AST from a stream of Token supplied by the Lexer. No semantic checks
/// or symbol resolution is performed. For example, variables are referenced by
/// string and the code could reference an undeclared variable and the parsing
/// succeeds.
class Parser {
public:
  /// Create a Parser for the supplied lexer.
  Parser(Lexer &lexer) : lexer(lexer) {}

  /// Parse a full Module. A module is a list of function definitions.
  std::unique_ptr<ModuleAST> parseModule() {
    lexer.getNextToken(); // prime the lexer

    // Parse functions one at a time and accumulate in this vector.
    std::vector<FunctionAST> functions;
    while (auto f = parseDefinition()) {
      functions.push_back(std::move(*f));
      if (lexer.getCurToken() == tok_eof)
        break;
    }
    // If we didn't reach EOF, there was an error during parsing
    if (lexer.getCurToken() != tok_eof)
      return parseError<ModuleAST>("nothing", "at end of module");

    return std::make_unique<ModuleAST>(std::move(functions));
  }

private:
  Lexer &lexer;

  /// Parse a function definition, we expect a prototype initiated with the
  /// `def` keyword, followed by a block containing a list of expressions.
  ///
  /// definition ::= prototype block
  std::unique_ptr<FunctionAST> parseDefinition() {
    auto proto = parsePrototype();
    if (!proto)
      return nullptr;

    if (auto block = parseBlock())
      return std::make_unique<FunctionAST>(std::move(proto), std::move(block));
    return nullptr;
  }

  // Parse a function prototype, which is represented as:
  //        prototype ::= def id '(' decl_list ')'   
  //        decl_list ::= identifier | identifier, decl_list                    
  std::unique_ptr<PrototypeAST> parsePrototype() {
    auto loc = lexer.getLastLocation();

    if (lexer.getCurToken() != tok_def)
      return parseError<PrototypeAST>("def", "in prototype");
    lexer.consume(tok_def);

    if (lexer.getCurToken() != tok_identifier)
      return parseError<PrototypeAST>("function name", "in prototype");

    std::string fnName(lexer.getId());
    lexer.consume(tok_identifier);

    if (lexer.getCurToken() != '(')
      return parseError<PrototypeAST>("(", "in prototype");
    lexer.consume(Token('('));

    std::vector<std::unique_ptr<VariableExprAST>> args;
    if (lexer.getCurToken() != ')') {
      do {
        std::string name(lexer.getId());
        auto loc = lexer.getLastLocation();
        lexer.consume(tok_identifier);
        auto decl = std::make_unique<VariableExprAST>(std::move(loc), name);
        args.push_back(std::move(decl));
        if (lexer.getCurToken() != ',')
          break;
        lexer.consume(Token(','));
        if (lexer.getCurToken() != tok_identifier)
          return parseError<PrototypeAST>(
            "identifier", "after ',' in function parameter list");
      } while (true);
    }

    if (lexer.getCurToken() != ')')
      return parseError<PrototypeAST>(")", "to end function prototype");

    // success.
    lexer.consume(Token(')'));

    return std::make_unique<PrototypeAST>(std::move(loc), fnName,
                                          std::move(args));
  }

  /// Parse a block: a list of expression separated by semicolons and wrapped in
  /// curly braces.
  ///
  /// block ::= { expression_list }
  /// expression_list ::= block_expr ; expression_list
  /// block_expr ::= decl | "return" | expr
  std::unique_ptr<ExprASTList> parseBlock() {
    if (lexer.getCurToken() != '{')
      return parseError<ExprASTList>("{", "to begin block");
    lexer.consume(Token('{'));

    auto exprList = std::make_unique<ExprASTList>();

    // Ignore empty expressions: swallow sequences of semicolons.
    while (lexer.getCurToken() == ';')
      lexer.consume(Token(';'));

    while (lexer.getCurToken() != '}' && lexer.getCurToken() != tok_eof) {
      // In pony_compiler, we focus on the implementation of variable declaration and general expression.
      if (lexer.getCurToken() == tok_var) {
        // Variable declaration
        auto varDecl = parseDeclaration();
        if (!varDecl)
          return nullptr;
        exprList->push_back(std::move(varDecl));
      } else if (lexer.getCurToken() == tok_return) {
        // Return statement
        auto ret = parseReturn();
        if (!ret)
          return nullptr;
        exprList->push_back(std::move(ret));
      } else {
        // General expression
        auto expr = parseExpression();
        if (!expr)
          return nullptr;
        exprList->push_back(std::move(expr));
      }
      // Ensure that elements are separated by a semicolon.
      if (lexer.getCurToken() != ';')
        return parseError<ExprASTList>(";", "after expression");

      // Ignore empty expressions: swallow sequences of semicolons.
      while (lexer.getCurToken() == ';')
        lexer.consume(Token(';'));
    }

    if (lexer.getCurToken() != '}')
      return parseError<ExprASTList>("}", "to close block");

    lexer.consume(Token('}'));
    return exprList;
  }

  // Parse a variable declaration, decl ::= var identifier [ type ] = expr
  // 1. it starts with a `var` keyword, followed by a variable name and initialization
  // 2. Two methods of initialization have been supported:
  //    (1) var a = [[1, 2, 3], [4, 5, 6]];
  //    (2) var a <2,3> = [1, 2, 3, 4, 5, 6];
  // You need to support the third method:
  //    (3) var a [2][3] = [1, 2, 3, 4, 5, 6];
  // Some functions may be useful:  getLastLocation(); getNextToken();
  std::unique_ptr<VarDeclExprAST> parseDeclaration() {
    auto loc = lexer.getLastLocation();
    std::string id;
    // TODO: check to see if this is a 'var' declaration 
    //       If not, report the error with 'parseError', otherwise eat 'var'  
    /* 
     *
     *  Write your code here.
     *
     */
    if (lexer.getCurToken() != tok_var)
      return parseError<VarDeclExprAST>("var", "to begin declaration");
    lexer.getNextToken();

    // TODO: check to see if this is a variable name(identifier)
    //       If not, report the error with 'parseError', otherwise eat the variable name
    /* 
     *
     *  Write your code here.
     *
     */
    if (lexer.getCurToken() != tok_identifier)
      return parseError<VarDeclExprAST>("identifier", "after 'var' declaration");
    id = lexer.getId().str();
    lexer.getNextToken();


    std::unique_ptr<VarType> type; // Type is optional, it can be inferred
    // TODO: modify the code to additionally support the third method: var a[][] = ... 
    if (lexer.getCurToken() == '<'|| lexer.getCurToken() == '[') {
      type = parseType();
      if (!type)
        return nullptr;
    }

    if (!type)
      type = std::make_unique<VarType>();
    lexer.consume(Token('='));
    auto expr = parseExpression();
    return std::make_unique<VarDeclExprAST>(std::move(loc), std::move(id),
                                            std::move(*type), std::move(expr));
  }

  /// type ::= < shape_list >
  /// shape_list ::= num | num , shape_list
  // TODO: make an extension to support the new type like var a[2][3] = [1, 2, 3, 4, 5, 6];
  std::unique_ptr<VarType> parseType() {
    Token cur = lexer.getCurToken();
    if (cur != '<' && cur != '[')
      return parseError<VarType>("< or [", "to begin type");

    auto type = std::make_unique<VarType>();

    if (cur == '<') {
      lexer.getNextToken(); // eat <
      while (lexer.getCurToken() == tok_number) {
        type->shape.push_back(lexer.getValue());
        lexer.getNextToken();
        if (lexer.getCurToken() == ',')
          lexer.getNextToken();
      }

      if (lexer.getCurToken() != '>')
        return parseError<VarType>(">", "to end type");

      lexer.getNextToken(); // eat >
    }
    else if (cur == '[') {
      while (lexer.getCurToken() == '[') {
        lexer.getNextToken();
        if (lexer.getCurToken() != tok_number) {
          return parseError<VarType>("<num>", "after '['");
        }
        type->shape.push_back(lexer.getValue());
        lexer.getNextToken();
        if (lexer.getCurToken() != ']')
          return parseError<VarType>("]", "to end type");
        lexer.getNextToken();
      }
    }
    return type;
  }

  /// Parse a return statement.
  /// return :== return ; | return expr ;
  std::unique_ptr<ReturnExprAST> parseReturn() {
    auto loc = lexer.getLastLocation();
    lexer.consume(tok_return);

    // return takes an optional argument
    llvm::Optional<std::unique_ptr<ExprAST>> expr;
    if (lexer.getCurToken() != ';') {
      expr = parseExpression();
      if (!expr)
        return nullptr;
    }
    return std::make_unique<ReturnExprAST>(std::move(loc), std::move(expr));
  }

  /// 解析函数内的表达式语句expression，其形式为：expression::= primary binop rhs
  std::unique_ptr<ExprAST> parseExpression() {
    /// 解析"="右边第一项：primary
    auto lhs = parsePrimary();
    if (!lhs)
      return nullptr;
    /// 解析剩余的项，可能有多种情况：
    /// 1. binop rhs
    /// 2. 无剩余项，直接返回primary对应的AST
    return parseBinOpRHS(0, std::move(lhs));
  }

  /// primary
  ///   ::= identifierexpr
  ///   ::= numberexpr
  ///   ::= parenexpr
  ///   ::= tensorliteral
  std::unique_ptr<ExprAST> parsePrimary() {
    switch (lexer.getCurToken()) {
      default:
        llvm::errs() << "unknown token '" << lexer.getCurToken()
          << "' when expecting an expression\n";
        return nullptr;
      // 解析标识符与函数调用，并返回相应AST
      case tok_identifier:
        return parseIdentifierExpr();
      // 解析数字，并返回相应AST
      case tok_number:
        return parseNumberExpr();
      // 解析括号表达式，并返回相应AST
      case '(':
        return parseParenExpr();
      // 解析Tensor的声明，并返回相应AST。
      case '[':
        return parseTensorLiteralExpr();
      case ';':
        return nullptr;
      case '}':
        return nullptr;
    }
  }

  // TODO: 解析标识符语句，其可以是简单的变量名，也可以用于函数调用。具有以下形式：
  //             ::= identifier
  //             ::= identifier '(' expression ')'
  // Hints: 1. 可以采用lexer中的适当方法获取当前的identifier，并eat identifier；
  //        2. 判断其为标识符，普通函数调用还是内置函数print的调用；
  //        3. 如果仅是一个变量名，则返回其对应的AST。可以使用std::make_unique<VariableExprAST>(...)；
  //        4. 如果是函数调用，其参数列表中的参数可以通过parseExpression()逐个解析，并存放在std::vector<std::unique_ptr<ExprAST>>中。最终返回相应AST时，可以使用std::make_unique<CallExprAST>(...)；
  //        5. 如果是print，要确保其内部只有一个参数，如果有多个参数，要输出错误信息。最终返回相应AST时，可以使用std::make_unique<PrintExprAST>(...)。
  std::unique_ptr<ExprAST> parseIdentifierExpr() {
    /* 
     *
     *  Write your code here.
     *
     */
    std::string callee(lexer.getId());
    auto loc = lexer.getLastLocation();

    if (lexer.getNextToken() != '(')
      return std::make_unique<VariableExprAST>(std::move(loc), callee);

    lexer.consume(Token('('));

    std::vector<std::unique_ptr<ExprAST>> args;
    if (lexer.getCurToken() != ')') {
      while (true) {
        if (auto arg = parseExpression())
          args.push_back(std::move(arg));
        else return nullptr;

        if (lexer.getCurToken() == ')') break;

        if (lexer.getCurToken() != ',')
          return parseError<ExprAST>(", or )", "in argument list");
        lexer.getNextToken();
      }
    }
    lexer.consume(Token(')'));

    if (callee == "print") {
      if (args.size() != 1)
        return parseError<ExprAST>("<single arg>", "as argument to print");
      return std::make_unique<PrintExprAST>(std::move(loc), std::move(args[0]));
    }

    return std::make_unique<CallExprAST>(std::move(loc), callee, std::move(args));
  }

  /// Parse a literal number.
  /// numberexpr ::= number
  std::unique_ptr<ExprAST> parseNumberExpr() {
    auto loc = lexer.getLastLocation();
    auto result =
      std::make_unique<NumberExprAST>(std::move(loc), lexer.getValue());
    lexer.consume(tok_number);
    return std::move(result);
  }

  // parse parenexpr ::= '(' expression ')'
  // The whole process can be divided into three steps: 1) eat '(' ; 2) parse 'expression' ; 3) eat ')' .
  std::unique_ptr<ExprAST> parseParenExpr() {
    lexer.getNextToken(); // eat (.
    auto v = parseExpression();
    if (!v)
      return nullptr;

    if (lexer.getCurToken() != ')')
      return parseError<ExprAST>(")", "to close expression with parentheses");
    lexer.consume(Token(')'));
    return v;
  }

  /// Parse a literal array expression.
  /// tensorLiteral ::= [ literalList ] | number
  /// literalList ::= tensorLiteral | tensorLiteral, literalList
  std::unique_ptr<ExprAST> parseTensorLiteralExpr() {
    auto loc = lexer.getLastLocation();
    lexer.consume(Token('['));

    // Hold the list of values at this nesting level.
    std::vector<std::unique_ptr<ExprAST>> values;
    // Hold the dimensions for all the nesting inside this level.
    std::vector<int64_t> dims;
    do {
      // We can have either another nested array or a number literal.
      if (lexer.getCurToken() == '[') {
        values.push_back(parseTensorLiteralExpr());
        if (!values.back())
          return nullptr; // parse error in the nested array.
      } else {
        if (lexer.getCurToken() != tok_number)
          return parseError<ExprAST>("<num> or [", "in literal expression");
        values.push_back(parseNumberExpr());
      }

      // End of this list on ']'
      if (lexer.getCurToken() == ']')
        break;

      // Elements are separated by a comma.
      if (lexer.getCurToken() != ',')
        return parseError<ExprAST>("] or ,", "in literal expression");

      lexer.getNextToken(); // eat ,
    } while (true);
    if (values.empty())
      return parseError<ExprAST>("<something>", "to fill literal expression");
    lexer.getNextToken(); // eat ]

    /// Fill in the dimensions now. First the current nesting level:
    dims.push_back(values.size());

    /// If there is any nested array, process all of them and ensure that
    /// dimensions are uniform.
    if (llvm::any_of(values, [](std::unique_ptr<ExprAST> &expr) {
                     return llvm::isa<LiteralExprAST>(expr.get());
                     })) {
      auto *firstLiteral = llvm::dyn_cast<LiteralExprAST>(values.front().get());
      if (!firstLiteral)
        return parseError<ExprAST>("uniform well-nested dimensions",
                                   "inside literal expression");

      // Append the nested dimensions to the current level
      auto firstDims = firstLiteral->getDims();
      dims.insert(dims.end(), firstDims.begin(), firstDims.end());

      // Sanity check that shape is uniform across all elements of the list.
      for (auto &expr : values) {
        auto *exprLiteral = llvm::cast<LiteralExprAST>(expr.get());
        if (!exprLiteral)
          return parseError<ExprAST>("uniform well-nested dimensions",
                                     "inside literal expression");
        if (exprLiteral->getDims() != firstDims)
          return parseError<ExprAST>("uniform well-nested dimensions",
                                     "inside literal expression");
      }
    }
    return std::make_unique<LiteralExprAST>(std::move(loc), std::move(values),
                                            std::move(dims));
  }

  // Get the precedence of the pending binary operator token.
  int getTokPrecedence() {
    if (!isascii(lexer.getCurToken()))
      return -1;

    // Currently we consider three binary operators: '+', '-', '*'.
    // Note that the smaller the number is, the lower precedence it will have. 
    switch (static_cast<char>(lexer.getCurToken())) {
      case '-':
        return 20;
      case '+':
        return 20;
      case '*':
        return 40;
      default:
        return -1;
    }
  }

  // TODO: Recursively parse the right hand side of a binary expression, 
  //       for example, it could be something like  '+' primary '-' primary  or  ('+' primary)* .
  //       The first argument (exprPrec) indicates the precedence of the current binary operator. 
  //       The second argument (lhs) indicates the left hand side of the expression.
  // Hints  1. Implement a recursive algorithm to parse;
  //        2. You may use some funtions in the lexer to get current and next tokens;
  //        3. You may use some functions to help you parse: getTokPrecedence(); parsePrimary(); parseError<ExprAST>();
  //        4. During each iteration, the lhs may be merged with rhs and becomes a larger lhs. Function you may use: std::make_unique<BinaryExprAST>(...).
  std::unique_ptr<ExprAST> parseBinOpRHS(int exprPrec,
                                         std::unique_ptr<ExprAST> lhs) {
    /* 
     *
     *  Write your code here.
     *
     */
    while (true) {
      int tokPrec = getTokPrecedence();

      if (tokPrec < exprPrec)
        return lhs;

      int binOp = lexer.getCurToken();
      lexer.consume(Token(binOp));
      auto loc = lexer.getLastLocation();

      auto rhs = parsePrimary();
      if (!rhs)
        return parseError<ExprAST>("expression", "to complete binary operator");

      int nextPrec = getTokPrecedence();
      if (nextPrec < tokPrec) {
        rhs = parseBinOpRHS(tokPrec + 1, std::move(rhs));
        if (!rhs)
          return nullptr;
      }

      lhs = std::make_unique<BinaryExprAST>(std::move(loc), binOp, std::move(lhs), std::move(rhs));
    }
  }

  /// Helper function to signal errors while parsing, it takes an argument
  /// indicating the expected token and another argument giving more context.
  /// Location is retrieved from the lexer to enrich the error message.
  template <typename R, typename T, typename U = const char *>
  std::unique_ptr<R> parseError(T &&expected, U &&context = "") {
    auto curToken = lexer.getCurToken();
    llvm::errs() << "Parse error (" << lexer.getLastLocation().line << ", "
      << lexer.getLastLocation().col << "): expected '" << expected
      << "' " << context << " but has Token " << curToken;
    if (isprint(curToken))
      llvm::errs() << " '" << (char)curToken << "'";
    llvm::errs() << "\n";
    return nullptr;
  }
};

} // namespace pony

#endif // PONY_PARSER_H
