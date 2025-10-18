import Foundation

let a: [Float] = [1, 2, 3, 4, 5]
let b: [Float] = [10, 20, 30, 40, 50]

do {
    let out = try MetalVectorAdder.add(a: a, b: b)
    print("GPU result:", out)
} catch {
    print("Metal error:", error.localizedDescription)
}
