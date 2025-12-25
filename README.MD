## ML OPS assignment 1 GROUP 113

#### after clone do the below:
1. python -m venv venv
2. source venv/bin/activate
3. pip install -r requirements.txt
##### TO RUN locally without Docker:
4. uvicorn api.main:app --host 127.0.0.1 --port 8080

##### TO RUN on docker
5. sudo docker build -t mlops-group113
6. sudo docker run -p 8080:8080 mlops-group113

note: replace the cloud run name and the region in mlops-pipeline yaml file and add the cloud run key to git hub secrets ith name GCP_SA_KEY