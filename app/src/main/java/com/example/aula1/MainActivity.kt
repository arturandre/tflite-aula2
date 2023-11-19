package com.example.aula1

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.util.Log
import android.widget.CompoundButton
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import com.example.aula1.databinding.ActivityMainBinding

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.common.TensorOperator
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageOperator
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.support.image.ops.TransformToGrayscaleOp
import java.lang.ClassCastException

// Initialization code

// Operations map
var opMap = mutableMapOf(
    "resize" to false,
    "crop" to false,
    "pad" to false,
    "rot" to false,
    "gray" to false,
    "norm" to false
)

val resOp = ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR)
val cropOp = ResizeWithCropOrPadOp(100, 100)
val padOp = ResizeWithCropOrPadOp(500, 500)
val rotOp = Rot90Op(2) // k*90 Anti-clockwise rotation
val grayOp = TransformToGrayscaleOp()
val normOp = NormalizeOp(127f, 1f)

var opMapF = mapOf(
    "resize" to resOp,
    "crop" to cropOp,
    "pad" to padOp,
    "rot" to rotOp,
    "gray" to grayOp,
    "norm" to normOp
)

/*var imageProcessor = ImageProcessor.Builder()
    .add(ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
    .build()*/

var imageProcessorBuilder: ImageProcessor.Builder? = null
var imageProcessor: ImageProcessor? = null
var bitmap: Bitmap? = null

// Create a TensorImage object. This creates the tensor of the corresponding
// tensor type (uint8 in this case) that the TensorFlow Lite interpreter needs.
var tensorImage: TensorImage? = TensorImage(DataType.UINT8)
var processedBitmap: Bitmap? = null


// Toggle buttons checked state change
fun opToggleButtonChangeListeners(bitmap: Bitmap, previewBitmap: ImageView):
        CompoundButton.OnCheckedChangeListener {
    return CompoundButton.OnCheckedChangeListener { buttonView, isChecked ->

        binding.previewBitmap.setImageBitmap(bitmap)
        //Log.d("MyTag", buttonView.tag as String + " " + isChecked)
        opMap[buttonView.tag as String] = isChecked
        tensorImage = TensorImage(DataType.UINT8)
        tensorImage?.load(bitmap)

        for ((key, value) in opMap) {
            if (value) {
                try {
                    imageProcessorBuilder = ImageProcessor.Builder()
                    imageProcessorBuilder?.add(opMapF[key] as ImageOperator)
                    imageProcessor = imageProcessorBuilder?.build()
                    tensorImage = imageProcessor?.process(tensorImage)
                } catch (e: ClassCastException) {
                    // Used for normalize operator
                    imageProcessorBuilder = ImageProcessor.Builder()
                    imageProcessorBuilder?.add(opMapF[key] as TensorOperator)
                    imageProcessor = imageProcessorBuilder?.build()
                    tensorImage = imageProcessor?.process(tensorImage)
                }

            }
        }
        // Processed TensorImage back into an bitmap
        processedBitmap = tensorImage?.bitmap
        binding.previewBitmap.setImageBitmap(processedBitmap)
    }
}


// Create an ImageProcessor with all ops required. For more ops, please
// refer to the ImageProcessor Architecture section in this README.


private lateinit var binding: ActivityMainBinding

class MainActivity : AppCompatActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        //Ref: https://developer.android.com/topic/libraries/view-binding
        binding = ActivityMainBinding.inflate(layoutInflater)
        val view = binding.root
        setContentView(view)

        // Sample image loader
        var bitmap = assets
            .open("cat1.png")
            .use(BitmapFactory::decodeStream)

        binding.previewBitmap.setImageBitmap(bitmap)

        // Loading sampling image into a TensorImage
        tensorImage?.load(bitmap)
        tensorImage = imageProcessor?.process(tensorImage)

        // Processed TensorImage back into an bitmap
        processedBitmap = tensorImage?.bitmap

        binding.resToggle.setOnCheckedChangeListener(
            opToggleButtonChangeListeners(
                bitmap,
                binding.previewBitmap
            )
        )
        binding.cropToggle.setOnCheckedChangeListener(
            opToggleButtonChangeListeners(
                bitmap,
                binding.previewBitmap
            )
        )
        binding.padToggle.setOnCheckedChangeListener(
            opToggleButtonChangeListeners(
                bitmap,
                binding.previewBitmap
            )
        )
        binding.rotToggle.setOnCheckedChangeListener(
            opToggleButtonChangeListeners(
                bitmap,
                binding.previewBitmap
            )
        )
        binding.grayToggle.setOnCheckedChangeListener(
            opToggleButtonChangeListeners(
                bitmap,
                binding.previewBitmap
            )
        )
        binding.normToggle.setOnCheckedChangeListener(
            opToggleButtonChangeListeners(
                bitmap,
                binding.previewBitmap
            )
        )
    }
}