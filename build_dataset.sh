 
##generate Corpus2 and Corpus3

cd dataset_builder/crawler

echo "Construct the corpus for recent Snopes news!"
mvn clean install
mvn exec:java -Dexec.mainClass=main.App -Dexec.args="mode3 Results/run000_snopes/ Snopes"

echo "Construct the corpus for recent Politifact news!"
mvn clean install
mvn exec:java -Dexec.mainClass=main.App -Dexec.args="mode3"

cd ../..	 

