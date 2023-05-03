package waig548.NNN.optimizer

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import waig548.NNN.util.math.Matrix
import waig548.NNN.util.math.plus
import waig548.NNN.util.math.times

@Serializable
@SerialName("Momentum")
class Momentum private constructor(
    val alpha: Double,
    val beta: Double,
    val momentDW: MutableList<Matrix<Double>>,
    val momentDB: MutableList<Matrix<Double>>
): Optimizer
{
    constructor(
        layerSizes: List<Pair<Int, Int>>,
        alpha: Double = 0.01,
        beta: Double = 0.9
    ): this(
        alpha, beta,
        MutableList(layerSizes.size) { Matrix(layerSizes[it].first, layerSizes[it].second) { 0.0 } },
        MutableList(layerSizes.size) { Matrix(layerSizes[it].second, 1) { 0.0 } },
    )
    override fun update(
        iteration: Int,
        layerID: Int,
        dW: Matrix<Double>,
        dB: Matrix<Double>
    ): Pair<Matrix<Double>, Matrix<Double>>
    {
        momentDW[layerID] = momentDW[layerID]*beta+dW*(1-beta)
        momentDB[layerID] = momentDB[layerID]*beta+dB*(1-beta)

        return momentDW[layerID]*alpha to momentDB[layerID]*alpha
    }
}