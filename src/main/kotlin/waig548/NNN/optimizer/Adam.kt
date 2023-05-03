package waig548.NNN.optimizer

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import waig548.NNN.util.math.Matrix
import waig548.NNN.util.math.div
import waig548.NNN.util.math.plus
import waig548.NNN.util.math.pow
import waig548.NNN.util.math.times
import kotlin.math.pow
import kotlin.math.sqrt

@Serializable
@SerialName("Adam")
class Adam private constructor(
    val alpha: Double,
    val beta1: Double,
    val beta2: Double,
    val epsilon: Double,
    val momentDW: MutableList<Matrix<Double>>,
    val momentDB: MutableList<Matrix<Double>>,
    val velocityDW: MutableList<Matrix<Double>>,
    val velocityDB: MutableList<Matrix<Double>>
) : Optimizer
{

    constructor(
        layerSizes: List<Pair<Int, Int>>,
        alpha: Double = 0.01,
        beta1: Double = 0.9,
        beta2: Double = 0.999,
        epsilon: Double = 1e-8
    ) : this(alpha, beta1, beta2, epsilon,
        MutableList(layerSizes.size) { Matrix(layerSizes[it].first, layerSizes[it].second) { 0.0 } },
        MutableList(layerSizes.size) { Matrix(layerSizes[it].second, 1) { 0.0 } },
        MutableList(layerSizes.size) { Matrix(layerSizes[it].first, layerSizes[it].second) { 0.0 } },
        MutableList(layerSizes.size) { Matrix(layerSizes[it].second, 1) { 0.0 } }
    )

    override fun update(
        iteration: Int,
        layerID: Int,
        //weights: Matrix<Double>,
        //biases: Matrix<Double>,
        dW: Matrix<Double>,
        dB: Matrix<Double>
    ): Pair<Matrix<Double>, Matrix<Double>>
    {
        momentDW[layerID] = momentDW[layerID]*beta1+dW*(1-beta1)
        momentDB[layerID] = momentDB[layerID]*beta1+dB*(1-beta1)
        velocityDW[layerID] = velocityDW[layerID]*beta2+pow(dW, 2.0)*(1-beta2)
        velocityDB[layerID] = velocityDB[layerID]*beta2+pow(dB, 2.0)*(1-beta2)

        val momentDWCorrection = momentDW[layerID]/(1-beta1.pow(sqrt(iteration.toDouble())))
        val momentDBCorrection = momentDB[layerID]/(1-beta1.pow(sqrt(iteration.toDouble())))
        val velocityDWCorrection = velocityDW[layerID]/(1-beta2.pow(sqrt(iteration.toDouble())))
        val velocityDBCorrection = velocityDB[layerID]/(1-beta2.pow(sqrt(iteration.toDouble())))

        return (div(momentDWCorrection, pow(velocityDWCorrection, 0.5)+epsilon))*alpha to
                (div(momentDBCorrection, pow(velocityDBCorrection, 0.5)+epsilon))*alpha
        //return (weights-div(momentDWCorrection, pow(velocityDWCorrection, 0.5))+epsilon) to
        //        (biases-div(momentDBCorrection, pow(velocityDBCorrection, 0.5))+epsilon)
    }

}