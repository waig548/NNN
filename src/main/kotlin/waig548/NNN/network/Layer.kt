package waig548.NNN.network

import kotlinx.serialization.Serializable
import kotlinx.serialization.Transient
import waig548.NNN.functions.ActivationFunction
import waig548.NNN.util.math.Matrix
import waig548.NNN.util.math.*
import waig548.NNN.util.math.checkDimensionExact
import waig548.NNN.util.math.div
import waig548.NNN.util.math.mul
import waig548.NNN.util.math.safeSoftMax
import waig548.NNN.util.math.scale
import waig548.NNN.util.math.sub
import waig548.NNN.util.math.times
import kotlin.random.Random

@Serializable
class Layer private constructor(
    val id: Int,
    var inputSize: Int,
    private val activationFunction: ActivationFunction,
    val neurons: List<Neuron>,
    var nextLayerID: Int? = null
) : NetworkBase()
{
    constructor(
        id: Int,
        random: Random,
        size: Int,
        inputSize: Int,
        activationFunction: ActivationFunction
    ) : this(
        id,
        inputSize,
        activationFunction,
        List(size) { Neuron(Matrix(inputSize) { random.nextDouble()*2-1.0 }, 0.0, activationFunction) })

    override val size: Int = neurons.size

    @Transient
    var nextLayer: Layer? = null

    @Transient
    var zs = Matrix(1) { 0.0 }

    @Transient
    val isLastLayer: Boolean = nextLayerID===null

    @Transient
    var scaleLength: Double = 0.0

    fun forward(input: Matrix<Double>): Matrix<Double>
    {
        x = input
        neurons.map { it.forward(input) }.unzip().apply {
            zs = Matrix.vectorOf(first)
            output = Matrix.vectorOf(second)
        }
        if (isLastLayer)
            safeSoftMax(zs).apply {
                output = first
                scaleLength = second
            }
        return output
    }

    fun backprop(diff: Matrix<Double>): Pair<Matrix<Double>, Pair<Matrix<Double>, Matrix<Double>>>
    {
        var db = mul(activationFunction.derivative(zs), scale(diff, 2.0))
        if (isLastLayer)
            db = diff * scaleLength
        val dW = ((x.transposed()(db))/x.dimension.toDouble()).transposed()
        val da = (Matrix(
            neurons[0].weight.dimension,
            neurons.size,
            neurons.map { it.weight }.reduce(Iterable<Double>::plus).toMutableList()
        ).transposed()(db.transposed())).transposed()
        return da to (dW to db)
    }

    fun updateParams(dW: Matrix<Double>, dB: Matrix<Double>)
    {
        val weights = Matrix.matrixOf(neurons.map { it.weight.toList() })
        val biases = Matrix.vectorOf(neurons.map { it.bias })
        checkDimensionExact(dW, weights)
        checkDimensionExact(dB, biases)
        val nW = sub(weights, dW).chunked(weights.width) { Matrix.vectorOf(it) }
        val nB = sub(biases, dB).toList()
        neurons.forEachIndexed { index, neuron ->
            neuron.weight = nW[index]
            neuron.bias = nB[index]
        }
    }

    override fun equals(other: Any?): Boolean
    {
        if (this===other) return true
        if (javaClass!=other?.javaClass) return false

        other as Layer

        if (neurons!=other.neurons) return false

        return true
    }

    override fun hashCode(): Int
    {
        return neurons.hashCode()
    }
}