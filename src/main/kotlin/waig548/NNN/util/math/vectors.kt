package waig548.NNN.util.math

import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.pow
import kotlin.math.sqrt

fun <T> checkDimensionExact(a: Matrix<T>, b: Matrix<T>): Boolean
{
    if (a.width!=b.width || a.height!=b.height)
        throw IllegalArgumentException("Matrix dimensions not match")
    return true
}

fun <T> checkDimensionMatrixMul(a: Matrix<T>, b: Matrix<T>): Boolean
{
    if (a.width!=b.height)
        throw IllegalArgumentException("Matrix dimensions not compatible")
    return true
}

fun add(a: Matrix<Double>, b: Matrix<Double>): Matrix<Double>
{
    checkDimensionExact(a, b)
    return Matrix(a.width, a.height, a.zip(b).map { it.first+it.second }.toMutableList())
}

fun sub(a: Matrix<Double>, b: Matrix<Double>): Matrix<Double>
{
    checkDimensionExact(a, b)
    return Matrix(a.width, a.height, a.zip(b).map { it.first-it.second }.toMutableList())
}

fun mul(a: Matrix<Double>, b: Matrix<Double>): Matrix<Double>
{
    checkDimensionExact(a, b)
    return Matrix(a.width, a.height, a.zip(b).map { it.first*it.second }.toMutableList())
}

fun div(a: Matrix<Double>, b: Matrix<Double>): Matrix<Double>
{
    checkDimensionExact(a, b)
    return Matrix(a.width, a.height, a.zip(b).map { it.first/it.second }.toMutableList())
}

fun pow(a: Matrix<Double>, b: Double): Matrix<Double>
{
    return Matrix(a.width, a.height, a.map { it.pow(b) }.toMutableList())
}

fun scale(a: Matrix<Double>, b: Double): Matrix<Double>
{
    return Matrix(a.width, a.height, a.map { it*b }.toMutableList())
}

fun dot(a: List<Double>, b: List<Double>): Double = a.zip(b).sumOf { it.first*it.second }

fun dot(a: Matrix<Double>, b: Matrix<Double>): Double
{
    checkDimensionExact(a, b)
    return a.zip(b).sumOf { it.first*it.second }
}

fun matrixMul(a: Matrix<Double>, b: Matrix<Double>): Matrix<Double>
{
    checkDimensionMatrixMul(a, b)
    return Matrix(
        b.width,
        a.height,
        a.chunked(a.width).map { i -> b.transposed().chunked(b.height).map { j -> dot(i, j) } }
            .reduce(Iterable<Double>::plus).toMutableList()
    )
}

fun applyFunction(function: ((Double) -> Double), a: Matrix<Double>): Matrix<Double>
{
    return Matrix(a.width, a.height, a.map(function).toMutableList())
}

fun clamp(a: Matrix<Double>, lowerBound: Double, upperBound: Double): Matrix<Double>
{
    return Matrix(a.width, a.height, a.map { min(max(lowerBound, it), upperBound) }.toMutableList())
}

fun normalize(vec: Matrix<Double>): Pair<Matrix<Double>, Pair<Double, Double>>
{
    val mean = vec.average()
    val variance = vec.reduce { acc, d -> acc+(d-mean)*(d-mean) }
    val stdDev = sqrt(variance/vec.dimension)
    return Matrix(vec.width, vec.height, vec.map { (it-mean)/stdDev }.toMutableList()) to (mean to stdDev)
}

fun scalingNormalize(vec: Matrix<Double>): Pair<Matrix<Double>, Double>
{
    val variance = vec.reduce { acc, d -> acc+d*d }
    val length = sqrt(variance)
    return Matrix(vec.width, vec.height, vec.map { it/length }.toMutableList()) to length
}

fun safeSoftMax(vec: Matrix<Double>): Pair<Matrix<Double>, Double>
{
    val (scaled, length) = scalingNormalize(vec)
    val expVec = applyFunction({ d -> exp(d)/*.takeIf { it>=1e-50 }  ?: 0.0 */ }, scaled)
    return expVec/expVec.sum() to length
}

fun softMax(vec: Matrix<Double>): Matrix<Double>
{
    //TODO("Fix error-prone softmax")
    val tmp = applyFunction({ if (it.isNaN()) 0.0 else it }, applyFunction(::exp, vec))
    return applyFunction({ if (it.isNaN()) 0.0 else it }, tmp/tmp.sum())
}

