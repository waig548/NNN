import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.decodeFromStream
import kotlinx.serialization.json.encodeToStream
import org.jetbrains.kotlin.konan.file.use
import org.junit.jupiter.api.Test
import waig548.NNN.functions.ReLU
import waig548.NNN.network.Network
import waig548.NNN.util.data.NetworkSerializer
import waig548.NNN.util.math.Matrix
import waig548.NNN.util.math.div
import waig548.NNN.util.math.minus
import waig548.NNN.util.math.mul
import waig548.NNN.util.math.plus
import waig548.NNN.util.math.pow
import java.io.FileInputStream
import java.io.FileOutputStream
import java.util.zip.GZIPInputStream
import java.util.zip.GZIPOutputStream
import kotlin.random.Random

class MNIST
{
    @Test
    fun `1-GenerateModel`()
    {
        val model = Network("MNIST", Random(System.currentTimeMillis()), 28*28, listOf(30, 30, 10), ReLU)
        NetworkSerializer.serializeToFile(model, "mnist")
    }

    @Test
    fun `2-SanitizeData`()
    {
        val rawTrainImages = GZIPInputStream(FileInputStream("mnist/train-images-idx3-ubyte.gz")).use { it.readBytes() }
        val rawTrainLabels = GZIPInputStream(FileInputStream("mnist/train-labels-idx1-ubyte.gz")).use { it.readBytes() }
        val rawTestImages = GZIPInputStream(FileInputStream("mnist/t10k-images-idx3-ubyte.gz")).use { it.readBytes() }
        val rawTestLabels = GZIPInputStream(FileInputStream("mnist/t10k-labels-idx1-ubyte.gz")).use { it.readBytes() }

        val train = rawTrainImages.drop(16)
            .map { it.toUByte().toDouble()/255.0 }
            .chunked(28*28)
            .map { Matrix(28*28, 1, it.toMutableList()) }
            .zip(rawTrainLabels.drop(8).map { List(10) { index -> if (index==it.toInt()) 1.0 else 0.0 } })

        val trainSet = DataSet(
            "MNIST-train",
            train.size,
            train.map { DataSet.Entry(it.first, Matrix(10, 1, it.second.toMutableList())) }
        )
        DataSetSerializer.serializeToFile(trainSet, "mnist")

        val test = rawTestImages.drop(16)
            .map { it.toUByte().toDouble()/255.0 }
            .chunked(28*28)
            .map { Matrix(28*28, 1, it.toMutableList()) }
            .zip(rawTestLabels.drop(8).map { List(10) { index -> if (index==it.toInt()) 1.0 else 0.0 } })
        val testSet = DataSet(
            "MNIST-test",
            test.size,
            test.map { DataSet.Entry(it.first, Matrix(10, 1, it.second.toMutableList())) }
        )
        DataSetSerializer.serializeToFile(testSet, "mnist")
    }

    @Test
    fun `3-Train`()
    {
        val model = NetworkSerializer.deserializeFromFile("mnist/MNIST.model")
        val trainSet = DataSetSerializer.deserializeFromFile("mnist/MNIST-train.data").shuffled().chunked(model.batchSize)
        val batches = trainSet.size
        val testSet = DataSetSerializer.deserializeFromFile("mnist/MNIST-test.data").shuffled().run { chunked(size/batches)}

        val iterator = trainSet.iterator()
        val testIterator = testSet.iterator()

        model.training = true
        while (iterator.hasNext())
        {
            val batch = iterator.next()
            val bi = batch.iterator()
            val d = MutableList(3) { emptyList<Pair<Matrix<Double>, Matrix<Double>>>().toMutableList() }
            val r = mutableListOf<Pair<Matrix<Double>, Matrix<Double>>>()
            while (bi.hasNext())
            {
                val next = bi.next()
                val tmp = model.backprop(next.input, next.output)
                r += model.output to next.output
                d.zip(tmp).forEach { it.first += it.second }
            }
            val avg = d.map {
                it.reduce { acc, pair -> acc.first+pair.first to acc.second+pair.second }.toList()
                    .map { m -> m/batch.size.toDouble() }.zipWithNext().first()
            }
            println(
                "Epoch: ${model.getEpoch()} of $batches, avg_diff: ${
                    Matrix.matrixOf(r.map {
                        pow(it.first-it.second, 2.0).toList()
                    }).transposed().chunked(batch.size).map { it.average() }
                }"
            )
            model.updateParams(avg)
            val testCases = testIterator.next()
            val result = testCases.map { model.iterate(it.input) }
            val isolated = result.zip(testCases.map { it.output }).map {
                mul(it.first, it.second).first { d -> d!=0.0 }
            }
            println("confidence: $isolated")

        }
        NetworkSerializer.serializeToFile(model, "mnist")
    }

    @Test
    fun `4-Test`()
    {
        val model = NetworkSerializer.deserializeFromFile("mnist/MNIST.model")
        val testSet = DataSetSerializer.deserializeFromFile("mnist/MNIST-test.data")
        val iterator = testSet.iterator()
        while (iterator.hasNext())
        {
            val case = iterator.next()
            val result = model.iterate(case.input)
            println(result)
        }
    }

    @Serializable
    data class DataSet(
        val name: String,
        override val size: Int,
        val data: List<Entry>
    ) : Collection<DataSet.Entry>
    {
        override fun contains(element: Entry): Boolean = data.contains(element)
        override fun containsAll(elements: Collection<Entry>): Boolean = data.containsAll(elements)
        override fun isEmpty(): Boolean = data.isEmpty()
        override fun iterator(): Iterator<Entry> = data.iterator()

        @Serializable
        data class Entry(
            val input: Matrix<Double>,
            val output: Matrix<Double>
        )
    }

    object DataSetSerializer
    {
        fun serializeToFile(obj: DataSet, path: String? = null)
        {
            GZIPOutputStream(FileOutputStream("${path?.let { "$it/" } ?: ""}${obj.name}.data")).use {
                Json.encodeToStream(obj, it)
            }
            println("Data saved to ${obj.name}.data")
        }

        fun deserializeFromFile(path: String): DataSet
        {
            val obj = GZIPInputStream(FileInputStream(path)).use { Json.decodeFromStream(DataSet.serializer(), it) }
            println("Data loaded from $path")
            return obj
        }
    }
}