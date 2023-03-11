import kotlinx.serialization.encodeToString
import kotlinx.serialization.json.Json
import org.junit.jupiter.api.Test
import waig548.NNN.functions.ReLU
import waig548.NNN.network.Network
import waig548.NNN.optimizer.Adam
import waig548.NNN.optimizer.Optimizer
import waig548.NNN.util.data.NetworkSerializer
import waig548.NNN.util.math.Matrix
import waig548.NNN.util.math.applyFunction
import waig548.NNN.util.math.matrixMul
import waig548.NNN.util.math.minus
import kotlin.math.abs
import kotlin.math.pow
import kotlin.random.Random

class Tests
{
    @Test
    fun `Test build save load model`()
    {
        val rand = Random(System.currentTimeMillis())
        val network = Network("Test1", rand, 10, listOf(5, 2, 3), ReLU)
        println(Json.encodeToString(network))
        NetworkSerializer.serializeToFile(network)
        val network2 = NetworkSerializer.deserializeFromFile("Test1.model")
        assert(network==network2)
    }

    @Test
    fun `Test Matrix`()
    {
        val m1 = Matrix(4, 3) { it+1.0 }
        println(Json.encodeToString(m1))
        val m2 = m1.transposed()
        println(Json.encodeToString(m2))
        val m3 = Matrix.I(4)
        println(Json.encodeToString(m3))
        val m4 = matrixMul(m2, m1)
        println(Json.encodeToString(m4))
        val m5 = Matrix(4, 3) { it+1.0 }
        println(m1==m5)
    }

    @Test
    fun `Test Serialize Optimizer`()
    {
        fun loss(m: Double) = m.pow(2)-2*m+1
        fun grad(m: Double) = 2*m-2
        fun converge(w0: Double, w1: Double) = abs(w0-w1)<1e-12

        val a1 = Adam(listOf(1 to 1))
        val s1 = Json.encodeToString(a1 as Optimizer)
        println(s1)
        val a2 = Json.decodeFromString(Optimizer.serializer(), s1)

        var w0 = Matrix.vectorOf(listOf(0.0))
        var b0 = Matrix.vectorOf(listOf(0.0))
        var t = 1
        var converged = false
        while (!converged)
        {
            val dw = applyFunction(::grad, w0)
            val db = applyFunction(::grad, b0)
            val w0_old = w0
            val tmp = a2.update(t, 0, dw, db)
            w0 -= tmp.first; b0 -= tmp.second
            if (converge(w0.first(), w0_old.first()))
            {
                println("Converged after $t iterations")
                converged = true
            } else
            {
                println("Iteration $t: weight = ${w0.first()}")
                t++
            }
        }
    }
}