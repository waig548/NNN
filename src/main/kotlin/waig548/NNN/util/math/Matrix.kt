package waig548.NNN.util.math

import kotlinx.serialization.Serializable
import kotlinx.serialization.Transient

@Serializable
class Matrix<T>(
    val width: Int,
    val height: Int,
    private val data: MutableList<T>
) : Iterable<T>
{
    constructor(
        width: Int,
        height: Int = 1,
        init: (index: Int) -> T
    ) : this(width, height, MutableList(width*height, init))
    //constructor(width: Int, height: Int, data: List<T>):this(width,height, data.toMutableList())

    @Transient
    val dimension = data.size

    fun get(x: Int, y: Int) = data[y*width+x]
    fun set(x: Int, y: Int, value: T)
    {
        data[y*width+x] = value
    }


    fun transposed() = Matrix(
        height,
        width,
        List(width) { i -> List(height) { j -> data[j*width+i] } }.reduce(Iterable<T>::plus).toMutableList()
    )

    override fun iterator(): Iterator<T> = data.iterator()
    override fun toString(): String
    {
        return "$width, $height, $data"
    }

    override fun equals(other: Any?): Boolean
    {
        if (this===other) return true
        if (javaClass!=other?.javaClass) return false

        other as Matrix<*>
        if (width!=other.width) return false
        if (height!=other.height) return false
        if (data!=other.data) return false
        return true
    }

    override fun hashCode(): Int
    {
        var result = width
        result = 31*result+height
        result = 31*result+data.hashCode()
        return result
    }

    companion object
    {
        fun <T> vectorOf(list: List<T>) = Matrix(list.size, 1, list::get)
        fun <T> matrixOf(matrix: List<List<T>>): Matrix<T>
        {
            if (matrix.isEmpty())
                throw IllegalArgumentException("List is empty")
            if (matrix.map { it.size }.distinct().size>1)
                throw IllegalArgumentException("Sizes of inner lists not identical")
            return Matrix(matrix.first().size, matrix.size, matrix.reduce(Iterable<T>::plus)::get)
        }

        fun I(dimension: Int): Matrix<Double> =
            Matrix(dimension, dimension) { index -> if (index%dimension==index/dimension) 1.0 else 0.0 }
    }
}