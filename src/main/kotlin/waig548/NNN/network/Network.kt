package waig548.NNN.network

import kotlinx.serialization.Serializable
import waig548.NNN.functions.ActivationFunction
import waig548.NNN.optimizer.Adam
import waig548.NNN.optimizer.Optimizer
import waig548.NNN.util.math.Matrix
import waig548.NNN.util.math.minus
import kotlin.random.Random

@Serializable
class Network private constructor(
    val name: String,
    private var inputSize: Int,
    private val activationFunction: ActivationFunction,
    val layers: MutableList<Layer>,
    val optimizer: Optimizer,
    private var epoch: Int
) : NetworkBase()
{
    constructor(
        name: String,
        rand: Random,
        inputSize: Int,
        layerSizes: List<Int>,
        activationFunction: ActivationFunction
    ) : this(
        name,
        inputSize,
        activationFunction,
        MutableList(layerSizes.size) {
            Layer(
                it,
                rand,
                layerSizes[it],
                if (it==0) inputSize else layerSizes[it-1],
                activationFunction
            )
        },
        Adam((layerSizes).mapIndexed { index: Int, size: Int -> (if (index==0) inputSize else layerSizes[index-1]) to size }),
        1
    )

    init
    {
        layers.forEachIndexed { index, layer ->
            if (index!=layers.size-1)
            {
                layer.nextLayer = layers[index+1]
                layer.nextLayerID = index+1
            }
        }
    }

    override val size: Int = layers.size

    fun getEpoch() = epoch

    fun iterate(input: Matrix<Double>): Matrix<Double>
    {
        var curData = input
        val iterator = layers.iterator()
        while (iterator.hasNext())
            curData = iterator.next().forward(curData)
        output = curData
        return curData
    }

    fun backprop(input: Matrix<Double>, desired: Matrix<Double>): List<Pair<Matrix<Double>, Matrix<Double>>>
    {

        val result = iterate(input)
        var diff = result-desired
        val iterator = layers.reversed().iterator()
        val d = mutableListOf<Pair<Matrix<Double>, Matrix<Double>>>()
        while (iterator.hasNext())
        {
            val tmp = iterator.next().backprop(diff)
            diff = tmp.first
            d.add(tmp.second)
        }
        return d.reversed()
    }

    fun updateParams(params: List<Pair<Matrix<Double>, Matrix<Double>>>)
    {
        val tmp = params.mapIndexed { index, pair -> optimizer.update(epoch, index, pair.first, pair.second) }
        layers.zip(tmp).forEach { it.first.updateParams(it.second.first, it.second.second) }
        epoch++
    }

    override fun equals(other: Any?): Boolean
    {
        if (this===other) return true
        if (javaClass!=other?.javaClass) return false

        other as Network

        if (name!=other.name) return false
        if (layers!=other.layers) return false

        return true
    }

    override fun hashCode(): Int
    {
        var result = name.hashCode()
        result = 31*result+layers.hashCode()
        return result
    }
}