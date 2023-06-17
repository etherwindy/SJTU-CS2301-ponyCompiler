/*===- TableGen'erated file -------------------------------------*- C++ -*-===*\
|*                                                                            *|
|* Dialect Declarations                                                       *|
|*                                                                            *|
|* Automatically generated file, do not edit!                                 *|
|*                                                                            *|
\*===----------------------------------------------------------------------===*/

namespace mlir {
namespace pony {

class PonyDialect : public ::mlir::Dialect {
  explicit PonyDialect(::mlir::MLIRContext *context)
    : ::mlir::Dialect(getDialectNamespace(), context,
      ::mlir::TypeID::get<PonyDialect>()) {
    
    initialize();
  }

  void initialize();
  friend class ::mlir::MLIRContext;
public:
  ~PonyDialect() override;
  static constexpr ::llvm::StringLiteral getDialectNamespace() {
    return ::llvm::StringLiteral("pony");
  }
};
} // namespace pony
} // namespace mlir
DECLARE_EXPLICIT_TYPE_ID(::mlir::pony::PonyDialect)
