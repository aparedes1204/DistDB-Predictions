package org.distributeddatabases.disastertweets;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.LogisticRegression;
import org.apache.spark.ml.feature.CountVectorizer;
import org.apache.spark.ml.feature.Tokenizer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;


public class Main {

	public static void main(String[] args) {
		
		// Start spark session
		SparkSession spark = SparkSession.builder()
				.appName("NLP with disaster Tweets").master("local[*]")
				.getOrCreate();
		
		spark.sparkContext().setLogLevel("ERROR");
		
		// List the data types to be used
		List<StructField> fields = Arrays.asList(
				DataTypes.createStructField("id", DataTypes.IntegerType, false),
				DataTypes.createStructField("keyword",  DataTypes.StringType, false),
				DataTypes.createStructField("location",  DataTypes.StringType, false),
				DataTypes.createStructField("text",  DataTypes.StringType, false),
				DataTypes.createStructField("target",  DataTypes.IntegerType, false));
		
		List<StructField> sub_fields = Arrays.asList(
				DataTypes.createStructField("id", DataTypes.IntegerType, false),
				DataTypes.createStructField("target",  DataTypes.IntegerType, false));
		
		// Create the needed schemas
		StructType schema = DataTypes.createStructType(fields);
		StructType sub_schema = DataTypes.createStructType(sub_fields);
		
		// Read the data form files
		Dataset<Row> train_df = spark.read().option("header", true).schema(schema).csv("src/main/resources/train.csv");
		Dataset<Row> test_df = spark.read().option("header", true).schema(schema).csv("src/main/resources/test.csv");
		Dataset<Row> sub_df = spark.read().option("header", true).schema(sub_schema).csv("src/main/resources/sample_submission.csv");
		
		// Clean null values
		train_df = train_df.na().drop(new String[] {"text", "target"}).withColumnRenamed("target", "label");
		test_df = test_df.na().drop(new String[] {"text"}).withColumnRenamed("target", "label");
		
		System.out.println("Train dataset");
		train_df.show(false);
		
		System.out.println("Test dataset");
		test_df.show(false);
		
		//  Set up Tokenizer to separate sentences into words
		Tokenizer tk = new Tokenizer().setInputCol("text").setOutputCol("words");
		
		// Set up CountVectorizer to count appearances of different words
		CountVectorizer cv = new CountVectorizer().setInputCol("words").setOutputCol("features").setMinDF(2);
		
		//Set up the Logistic regression algorithm for the classification
		LogisticRegression lr = new LogisticRegression().setElasticNetParam(0.0);
		
		//Set up the pipeline to save the models when predicting results
		Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] {tk,cv, lr});
		
		//Training the model
		PipelineModel pipelineModel = pipeline.fit(train_df);
		
		//Making predictions
		Dataset<Row> prediction = pipelineModel.transform(test_df).select("id", "prediction");
		
		sub_df = sub_df.join(prediction, "id").select("id", "prediction");
		System.out.println("Predictions");
		sub_df.show(false);
		
	}

}
