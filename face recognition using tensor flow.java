//facenet model

import org.tensorflow.Graph;  
import org.tensorflow.Session;  
import org.tensorflow.Tensor;  
import org.tensorflow.TensorFlow;  
  
public class FaceNetModel {  
  
    private Graph graph;  
    private Session session;  
    private String modelPath;  
  
    public FaceNetModel(String modelPath) {  
        this.modelPath = modelPath;  
    }  
  
    public void loadModel() {  
        graph = new Graph();  
        byte[] modelBytes = readAllBytesOrExit(Paths.get(modelPath));  
        graph.importGraphDef(modelBytes);  
        session = new Session(graph);  
    }  
  
    public float[] getFaceEmbedding(Mat face) {  
        float[] embedding = null;  
        try (Tensor<Float> tensor = normalizeImage(face)) {  
            Tensor<Float> output = session.runner()  
                    .feed("input_1", tensor)  
                    .fetch("Bottleneck_BatchNorm/batchnorm/add_1")  
                    .run()  
                    .get(0)  
                    .expect(Float.class);  
            embedding = new float[(int) output.shape()[1]];  
            output.copyTo(embedding);  
        } catch (Exception e) {  
            e.printStackTrace();  
        }  
        return embedding;  
    }  
  
    private Tensor<Float> normalizeImage(Mat mat) {  
        Imgproc.cvtColor(mat, mat, Imgproc.COLOR_BGR2RGB);  
        mat.convertTo(mat, CvType.CV_32F);  
        Core.divide(mat, Scalar.all(255.0f), mat);  
        return Tensor.create(mat.reshape(1, 160, 160, 3));  
    }  
  
    private static byte[] readAllBytesOrExit(Path path) {  
        try {  
            return Files.readAllBytes(path);  
        } catch (IOException e) {  
            e.printStackTrace();  
            System.exit(-1);  
        }  
        return null;  
    }  
  
}


///loading the face recognition database

private Map<String, float[]> faceDb = new HashMap<>();  
  
public void loadFaceDb(String dbPath) {  
    try {  
        BufferedReader reader = new BufferedReader(new FileReader(dbPath));  
        String line;  
        while ((line = reader.readLine()) != null) {  
            String[] values = line.split(",");  
            String name = values[0];  
            float[] embedding = Arrays.stream(values[1].split(" ")).map(Float::parseFloat).toArray(float[]::new);  
            faceDb.put(name, embedding);  
        }  
        reader.close();  
    } catch (IOException e) {  
        e.printStackTrace();  
    }  
} 


//implement face recognition

private FaceNetModel faceNetModel = new FaceNetModel("facenet.pb");  
  
public void recognizeFaces(Mat frame) {  
    Mat gray = new Mat();  
    Imgproc.cvtColor(frame, gray, Imgproc.COLOR_BGR2GRAY);  
    MatOfRect faces = new MatOfRect();  
    cascadeClassifier.detectMultiScale(gray, faces, 1.3, 5);  
  
    for (Rect rect : faces.toArray()) {  
        Mat face = new Mat(frame, rect);  
        Imgproc.resize(face, face, new Size(160, 160));  
        float[] embedding = faceNetModel.getFaceEmbedding(face);  
        String name = recognizeFace(embedding);  
        Imgproc.putText(frame, name, new Point(rect.x, rect.y - 10), Imgproc.FONT_HERSHEY_SIMPLEX, 1, new Scalar(0, 255, 0), 2);  
        Imgproc.rectangle(frame, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0), 2);  
    }  
}  
  
private String recognizeFace(float[] embedding) {  
    String name = "Unknown";  
    double minDistance = Double.MAX_VALUE;  
    for (Map.Entry<String, float[]> entry : faceDb.entrySet()) {  
        float[] dbEmbedding = entry.getValue();  
        double distance = calculateDistance(embedding, dbEmbedding);  
        if (distance < minDistance) {  
            minDistance = distance;  
            name = entry.getKey();  
        }  
    }  
    if (minDistance > threshold) {  
        name = "Unknown";  
    }  
    return name;  
}  
  
private double calculateDistance(float[] embedding1, float[] embedding2) {  
    double sum = 0.0;  
    for (int i = 0; i < embedding1.length; i++) {  
        sum += Math.pow(embedding1[i] - embedding2[i], 2);  
    }  
    return Math.sqrt(sum);  
}  


////This code defines a recognizeFaces method that takes a video frame as input, detects faces in the frame using a Haar Cascade classifier, encodes each face into an embedding using the FaceNet model, and compares the embeddings to those in the face recognition database using the calculateDistance method.
 


