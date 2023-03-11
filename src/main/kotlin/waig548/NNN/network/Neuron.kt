package waig548.NNN.network

import kotlinx.serialization.Serializable
import waig548.NNN.functions.ActivationFunction
import waig548.NNN.util.math.Matrix
import waig548.NNN.util.math.dot

@Serializable
class Neuron(
    var weight: Matrix<Double>,
    var bias: Double,
    val activationFunction: ActivationFunction
)
{
    fun forward(input: Matrix<Double>): Pair<Double, Double>
    {
        val z = dot(weight, input)+bias
        return z to activationFunction(z)
    }

    override fun equals(other: Any?): Boolean
    {
        if (this===other) return true
        if (javaClass!=other?.javaClass) return false

        other as Neuron

        if (weight!=other.weight) return false
        if (bias!=other.bias) return false
        if (activationFunction.javaClass!=other.activationFunction.javaClass) return false

        return true
    }

    override fun hashCode(): Int
    {
        var result = weight.hashCode()
        result = 31*result+bias.hashCode()
        result = 31*result+activationFunction.hashCode()
        return result
    }
}