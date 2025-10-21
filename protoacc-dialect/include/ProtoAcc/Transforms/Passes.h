#ifndef PROTOACC_TRANSFORMS_PASSES_H
#define PROTOACC_TRANSFORMS_PASSES_H

#include <memory>

namespace mlir {
class Pass;
} // namespace mlir

namespace protoacc {
std::unique_ptr<mlir::Pass> createLowerToProtoAccPass();
void registerProtoAccPasses();
} // namespace protoacc

#endif // PROTOACC_TRANSFORMS_PASSES_H
