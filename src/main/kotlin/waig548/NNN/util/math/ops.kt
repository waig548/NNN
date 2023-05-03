package waig548.NNN.util.math

operator fun Matrix<Double>.plus(b: Matrix<Double>): Matrix<Double>
{
    if (width == b.width && height!=b.height)
        return Matrix.matrixOf(chunked(b.width).chunked(b.height).map { (Matrix.matrixOf(it)+Matrix.matrixOf(b.chunked(it.first().size).slice(it.indices))).chunked(width) }.reduce(List<List<Double>>::plus))
    if (height == b.height && width!=b.width)
        return Matrix.matrixOf(chunked(width).zip(b.chunked(b.width)).map { (Matrix.vectorOf(it.first).transposed()+Matrix.vectorOf(it.second).transposed()).transposed().toList() })
    return add(this, b)
}
operator fun Matrix<Double>.plus(scale: Double) = add(this, Matrix(this.width, this.height) { scale })
operator fun Matrix<Double>.unaryPlus() = this
operator fun Matrix<Double>.minus(b: Matrix<Double>): Matrix<Double>
{
    if (width == b.width && height!=b.height)
        return Matrix.matrixOf(chunked(b.width).chunked(b.height).map { (Matrix.matrixOf(it)-Matrix.matrixOf(b.chunked(it.first().size).slice(it.indices))).chunked(width) }.reduce(List<List<Double>>::plus))
    if (height == b.height && width!=b.width)
        return Matrix.matrixOf(chunked(width).zip(b.chunked(b.width)).map { (Matrix.vectorOf(it.first).transposed()-Matrix.vectorOf(it.second).transposed()).transposed().toList() })
    return sub(this, b)
}
operator fun Matrix<Double>.minus(scale: Double) = sub(this, Matrix(this.width, this.height) { scale })
operator fun Matrix<Double>.unaryMinus() = this*-1.0
operator fun Matrix<Double>.times(scale: Double) = scale(this, scale)
operator fun Matrix<Double>.times(b: Matrix<Double>): Matrix<Double>
{
    if (width == b.width && height!=b.height)
        return Matrix.matrixOf(chunked(b.width).chunked(b.height).map { (Matrix.matrixOf(it)*Matrix.matrixOf(b.chunked(it.first().size).slice(it.indices))).chunked(width) }.reduce(List<List<Double>>::plus))
    if (height == b.height && width!=b.width)
        return Matrix.matrixOf(chunked(width).zip(b.chunked(b.width)).map { (Matrix.vectorOf(it.first).transposed()*Matrix.vectorOf(it.second).transposed()).transposed().toList() })
    return mul(this, b)
}
operator fun Matrix<Double>.div(scale: Double) = scale(this, 1.0/scale)
operator fun Matrix<Double>.div(b: Matrix<Double>): Matrix<Double>
{
    if (width == b.width && height!=b.height)
        return Matrix.matrixOf(chunked(b.width).chunked(b.height).map { (Matrix.matrixOf(it)/Matrix.matrixOf(b.chunked(it.first().size).slice(it.indices))).chunked(width) }.reduce(List<List<Double>>::plus))
    if (height == b.height && width!=b.width)
        return Matrix.matrixOf(chunked(width).zip(b.chunked(b.width)).map { (Matrix.vectorOf(it.first).transposed()/Matrix.vectorOf(it.second).transposed()).transposed().toList() })
    return div(this, b)
}
operator fun Matrix<Double>.invoke(b: Matrix<Double>) = matrixMul(this, b)
fun Matrix<Double>.sum(axis: Int): Matrix<Double>
{
    return Matrix.vectorOf(when (axis)
    {
        0 -> this
        1 -> transposed()
        else -> throw IllegalArgumentException("Axis must be either 0 or 1")
    }.chunked(shape[axis]).map(List<Double>::sum)).run { if (axis == 0) transposed() else this }
}
fun Matrix<Double>.mean(axis: Int): Matrix<Double>
{
    return sum(axis)/shape[axis].toDouble()
}
fun Matrix<Double>.variance(axis: Int): Matrix<Double>
{
    return  Matrix.vectorOf(when (axis)
    {
        0 -> this
        1 -> transposed()
        else -> throw IllegalArgumentException("Axis must be either 0 or 1")
    }.chunked(shape[axis]).zip(mean(axis)).map { pow(Matrix.vectorOf(it.first)-it.second, 2.0).average() }).run { if (axis == 0) transposed() else this }
}
operator fun ((Double) -> Double).invoke(matrix: Matrix<Double>): Matrix<Double> = applyFunction(this, matrix)