package skymind.ai.midterm;

import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.collection.CollectionRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.filter.FilterInvalidValues;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.util.ndarray.RecordConverter;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.earlystopping.scorecalc.DataSetLossCalculator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.BatchNormalization;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.ViewIterator;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.MinMaxSerializerStrategy;
import org.nd4j.linalg.dataset.api.preprocessor.serializer.NormalizerSerializer;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.schedule.MapSchedule;
import org.nd4j.linalg.schedule.ScheduleType;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;

public class question1termdeposit {
    //term deposit classifier

//    private static int numLinesToSkip = 0; //no line to skip
//    private static char delimiter = ',';

    //1. Setting the variables
    final private static int seed = 1234;
    final private static int batchSize = 1000;
    final private static int epoch = 5;


//    private static int batchSize = 1000; // data set:
//    private static int labelIndex = 17; // index of label/class column (correspond to the features
//    private static int numClasses = 2; // number of class in iris dataset (In the last column = 0,1)
//    private static File inputFile;

    //2. File location and get file
    public static void main(String[] args) throws IOException, InterruptedException {

        File trainFilePath = new ClassPathResource("termdeposit/train.csv").getFile();

        //2. Read file
        CSVRecordReader trainCSVrr = new CSVRecordReader(1, ',');
        trainCSVrr.initialize(new FileSplit(trainFilePath));

        //3. Define schema (get schema)
        Schema trainSchema = getTrainSchema();

        //4. Transform process
        TransformProcess trainTP = new TransformProcess.Builder(trainSchema)
                //during transform process it cannot accept categorical things
                //convert categorical to integer
                .categoricalToInteger("job", "marital", "education", "default", "housing", "loan",
                        "contact", "month", "poutcome", "subscribed")
                .filter(new FilterInvalidValues()) //filter off the null values
                .build();

        //5. Keep all data into collection
        List<List<Writable>> oriData = new ArrayList();

        while (trainCSVrr.hasNext()) {
            oriData.add(trainCSVrr.next());
        }

        //6. Do transform process
        List<List<Writable>> trainTransformedData = LocalTransformExecutor.execute(oriData, trainTP);

        //7. print out final schema, size of data and transform size
        System.out.println(trainTP.getFinalSchema());
        System.out.println(oriData.size());
        System.out.println(trainTransformedData.size());

        //8. Reading collection using CollectionRecordReader
        CollectionRecordReader cRR = new CollectionRecordReader(trainTransformedData);
        //throw into record reader data set iterator
        RecordReaderDataSetIterator dataIter = new RecordReaderDataSetIterator(cRR, trainTransformedData.size(),
                17, 2);
        DataSet dataSet = dataIter.next();
        //dataSet.setLabelNames(Arrays.asList("0", "1"));

        //9. split, test & train
        SplitTestAndTrain split = dataSet.splitTestAndTrain(0.8);
        DataSet trainSplit = split.getTrain();
        DataSet valSplit = split.getTest();
        //10. set label names
        trainSplit.setLabelNames(Arrays.asList("0", "1"));
        valSplit.setLabelNames(Arrays.asList("0","1"));

        //11. Normalization
        NormalizerMinMaxScaler scaler = new NormalizerMinMaxScaler();
        scaler.fit(trainSplit);
        scaler.transform(trainSplit);
        scaler.transform(valSplit);

        //12. put both into iterator - minibatch
        ViewIterator trainIter = new ViewIterator((org.nd4j.linalg.dataset.DataSet) trainSplit, batchSize);
        ViewIterator testIter = new ViewIterator((org.nd4j.linalg.dataset.DataSet) valSplit, batchSize);

        //13. use hashmap and scheduler for learning rate schedule
        //hashmaps associate integer to double
        HashMap<Integer, Double> scheduler = new HashMap<>();
        scheduler.put(0, 1e-3); //0-> epoch, 1e-3 -> learning rate
        scheduler.put(2, 1e-4);
        scheduler.put(3, 1e-5);

        //14. config model
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(new MapSchedule(ScheduleType.EPOCH, scheduler)))
                .weightInit(WeightInit.XAVIER)
                .activation(Activation.RELU)
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(trainIter.inputColumns())
                        .nOut(256)
                        .build())
                .layer(1, new BatchNormalization())
                .layer(2, new DenseLayer.Builder()
                        .nOut(512)
                        .build())
                .layer(3, new BatchNormalization())
                .layer(4, new DenseLayer.Builder()
                        .nOut(512)
                        .build())
                .layer(5, new BatchNormalization())
                .layer(6, new DenseLayer.Builder()
                        .nOut(512)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.XENT)
                        .activation(Activation.SIGMOID)
                        //.activation(Activation.SIGMOID)
                        .nOut(trainIter.totalOutcomes())
                        .build())
                .build();

        //15. multilayer model
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        InMemoryStatsStorage storage = new InMemoryStatsStorage();
        //16. monitor progress - use listeners
        //get place to store value
        //get server -train UIserver
        UIServer server = UIServer.getInstance();
        server.attach(storage);
        //set listeners - help to track training process
        //need router - storage & frequency
        //UI from browser
        //score listener from console
        //set listener - listen to model - tell what is the loss
        model.setListeners(new StatsListener(storage), new ScoreIterationListener(1000)); //100

        //17. Check lost
        ArrayList<Double> trainLoss = new ArrayList<>();
        ArrayList<Double> valLoss = new ArrayList<>();
        DataSetLossCalculator trainLossCalc = new DataSetLossCalculator(trainIter, true);
        DataSetLossCalculator testLossCalc = new DataSetLossCalculator(testIter, true);

        for (int i = 0; i < epoch; i++) {
            model.fit(trainIter);
            trainLoss.add(trainLossCalc.calculateScore(model));
            valLoss.add(testLossCalc.calculateScore(model));

        }

        //18. evaluation
        Evaluation trainEval = model.evaluate(trainIter);
        Evaluation testEval = model.evaluate(testIter);

        System.out.println(trainEval.stats());
        System.out.println(testEval.stats());

        //19. *IMPORTANT PART*
        //saving model and rename as termdepo1.zip
        //extension for DL4J - zip file
        ModelSerializer.writeModel(model,"C:\\Users\\ASUS\\Desktop\\model\\termdepo1.zip", true);
        //for serializer
        //saving statistics for normalizer
        NormalizerSerializer normalizerSerializer = new NormalizerSerializer().addStrategy(new MinMaxSerializerStrategy());
        normalizerSerializer.write(scaler, "C:\\Users\\ASUS\\Desktop\\model\\normalizer.zip");

        //20. *IMPORTANT FOR CPU*
        Nd4j.getEnvironment().allowHelpers(false);
        //convert dataset to INDArray
        //convert to collection
        List<List<Writable>> valCollection = RecordConverter.toRecords(valSplit);
        //convert to arrays
        INDArray valArray = RecordConverter.toMatrix(DataType.FLOAT, valCollection);
        //don't want label, just features
        INDArray valFeatures = valArray.getColumns(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16);

        //supply and set labels
        List<String> prediction = model.predict(valSplit);
        INDArray output = model.output(valFeatures);

        for (int i = 0; i < 10; i++) {
            System.out.println("Prediction:" + prediction.get(i) + "; Output: " + output.getRow(i));
        }

    }

    static Schema getTrainSchema(){

        return new Schema.Builder()
                .addColumnInteger("ID")
                .addColumnInteger("age")
                .addColumnCategorical("job", Arrays.asList("admin.", "services", "management", "technician",
                        "retired", "blue-collar", "housemaid", "student", "entrepreneur", "self-employed",
                        "unemployed", "unknown"))
                .addColumnCategorical("marital", Arrays.asList("married", "divorced", "single"))
                .addColumnCategorical("education", Arrays.asList("unknown", "secondary", "tertiary", "primary"))
                .addColumnCategorical("default", Arrays.asList("no", "yes"))
                .addColumnDouble("balance")
                .addColumnCategorical("housing", Arrays.asList("no", "yes"))
                .addColumnCategorical("loan", Arrays.asList("no", "yes"))
                .addColumnCategorical("contact", Arrays.asList("telephone", "cellular", "unknown"))
                .addColumnInteger("day")
                .addColumnCategorical("month", Arrays.asList("jan", "feb", "mar", "apr", "may", "jun",
                        "jul", "aug", "sep", "oct", "nov", "dec"))
                .addColumnInteger("duration")
                .addColumnInteger("campaign")
                .addColumnInteger("pdays")
                .addColumnInteger("previous")
                .addColumnCategorical("poutcome", Arrays.asList("unknown", "success", "failure", "other"))
                .addColumnCategorical("subscribed", Arrays.asList("no", "yes"))
                .build();

    }

}
