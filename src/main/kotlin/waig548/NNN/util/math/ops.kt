package waig548.NNN.util.math

operator fun Matrix<Double>.plus(b: Matrix<Double>) = add(this, b)
operator fun Matrix<Double>.plus(scale: Double) = add(this, Matrix(this.width, this.height) { scale })
operator fun Matrix<Double>.unaryPlus() = this
operator fun Matrix<Double>.minus(b: Matrix<Double>) = sub(this, b)
operator fun Matrix<Double>.minus(scale: Double) = sub(this, Matrix(this.width, this.height) { scale })
operator fun Matrix<Double>.unaryMinus() = this*-1.0
operator fun Matrix<Double>.times(scale: Double) = scale(this, scale)
operator fun Matrix<Double>.div(scale: Double) = scale(this, 1.0/scale)
operator fun Matrix<Double>.times(b: Matrix<Double>) = matrixMul(this, b)
