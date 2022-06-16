 
##generate Corpus2 and Corpus3

cd dataset_builder/crawler

echo "Construct the corpus2 and Corpus3 for recent Snopes news!"
mvn clean install
mvn exec:java -Dexec.mainClass=main.App -Dexec.args="mode3 Results/run000_snopes/ Snopes"

echo "Construct the corpus2 and Corpus3 for recent Politifact news!"
mvn clean install
mvn exec:java -Dexec.mainClass=main.App -Dexec.args="mode3"

cd ../..	 

##generate images folder
echo "Construct the images folder for recent Politifact news!"
cd dataset_builder/multi_media
python fetch_img.py 
cd ../..	 

##