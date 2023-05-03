package waig548.NNN.normalization

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.Transient
import waig548.NNN.optimizer.Momentum
import waig548.NNN.util.math.Matrix
import waig548.NNN.util.math.*

@Serializable
@SerialName("BN")
class BatchNormalization private constructor(
    val batchSize: Int,
    val gamma: MutableList<Matrix<Double>>,
    val beta: MutableList<Matrix<Double>>,
    val epsilon: Double,
    val momentum: Momentum
): Normalization
{
    constructor(
        layerSizes: List<Int>,
        batchSize: Int,
        epsilon: Double = 1e-8
    ): this(
        batchSize,
        MutableList(layerSizes.size) { Matrix(layerSizes[it]) { 1.0 } },
        MutableList(layerSizes.size) { Matrix(layerSizes[it]) { 0.0 } },
        epsilon,
        Momentum(layerSizes.zip(List(layerSizes.size){1}), 1.0)
    )

    @Transient
    val cache: MutableList<List<Matrix<Double>>> = MutableList(gamma.size){ listOf() }
    override fun normalize(x: Matrix<Double>, layerID: Int, training: Boolean): Matrix<Double>
    {
        val out: Matrix<Double>
        if(training)
        {
            val mean = x.mean(1)
            val variance = x.variance(1)
            val (rVar, rMean) = momentum.update(0, layerID, variance, mean)
            val stdDev = pow(variance+epsilon, 0.5)
            val xCentered = x - rMean
            val xNorm = xCentered / stdDev

            cache[layerID] = listOf(xNorm, xCentered, stdDev, gamma[layerID])
            out = gamma[layerID] * xNorm + beta[layerID]
        }
        else
        {
            val xNorm = (x - momentum.momentDB[layerID])/pow(momentum.momentDW[layerID]+epsilon, 0.5)
            out =  gamma[layerID] * xNorm + beta[layerID]
        }
        return out
    }

    override fun update(dy: Matrix<Double>, layerID: Int): Matrix<Double>
    {
        TODO("Not yet implemented")
    }
}