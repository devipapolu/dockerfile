from pymongo import MongoClient
import yaml

# Load configuration
with open(r"config.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# Connect to MongoDB Atlas
connection_str = config['db_connection_str']
client = MongoClient(connection_str)

# Access the database and collections
MONGO_DB_NAME = config['MONGO_DB_NAME']
db = client[MONGO_DB_NAME]

# Collections based on config
employees_collection = db[config['facedb_collection_name']]  # Collection for employees
visitors_collection = db[config['visitor_collection_name']]  # Collection for visitors
attendance_logs_collection = db[config['attendance_collection_name']]  # Collection for attendance logs
alerts_collection = db.get(config.get('alert_collection_name', 'alerts'))  # Optional alerts collection

def get_registered_employees_info():
    """
    Retrieves all registered employee embeddings, names, genders, and person IDs from the database.
    Supports multiple embeddings per employee.
    """
    embeddings = []
    person_ids = []
    names = []
    genders = []

    for record in employees_collection.find():
        person_id = record.get("person_id")
        name = record.get("name")
        gender = record.get("gender")
        person_embeddings = record.get("embeddings", [])

        # Ensure that required fields are present
        if person_id and name and gender and person_embeddings:
            for embedding_entry in person_embeddings:
                embedding = embedding_entry.get("embedding")
                if embedding:  # Validate the embedding
                    embeddings.append(embedding)
                    person_ids.append(person_id)
                    names.append(name)
                    genders.append(gender)

    # Ensure valid data before returning
    if embeddings and person_ids and names and genders:
        return embeddings, person_ids, names, genders
    else:
        raise ValueError("No valid employee face data found in the database.")

def get_registered_visitors_info():
    """
    Retrieves all registered visitor embeddings, names, purposes, and visitor IDs from the database.
    Supports multiple embeddings per visitor.
    """
    embeddings = []
    visitor_ids = []
    names = []
    purposes = []

    for record in visitors_collection.find():
        visitor_id = record.get("visitor_id")
        name = record.get("name")
        visit_purpose = record.get("visit_purpose")
        person_embeddings = record.get("embeddings", [])

        # Ensure that required fields are present
        if visitor_id and name and visit_purpose and person_embeddings:
            for embedding_entry in person_embeddings:
                embedding = embedding_entry.get("embedding")
                if embedding:  # Validate the embedding
                    embeddings.append(embedding)
                    visitor_ids.append(visitor_id)
                    names.append(name)
                    purposes.append(visit_purpose)

    # Ensure valid data before returning
    if embeddings and visitor_ids and names and purposes:
        return embeddings, visitor_ids, names, purposes
    else:
        raise ValueError("No valid visitor face data found in the database.")

def get_registered_onboarding_info():
    """
    Retrieves all registered onboarding person embeddings, names, purposes, and IDs from the database.
    Supports multiple embeddings per onboarding person.
    """
    embeddings = []
    onboarding_ids = []
    names = []
    purposes = []

    for record in db[config['onboarding_collection_name']].find():
        onboarding_id = record.get("onboarding_id")
        name = record.get("name")
        onboarding_purpose = record.get("onboarding_purpose")
        person_embeddings = record.get("embeddings", [])

        # Ensure that required fields are present
        if onboarding_id and name and onboarding_purpose and person_embeddings:
            for embedding_entry in person_embeddings:
                embedding = embedding_entry.get("embedding")
                if embedding:  # Validate the embedding
                    embeddings.append(embedding)
                    onboarding_ids.append(onboarding_id)
                    names.append(name)
                    purposes.append(onboarding_purpose)

    # Ensure valid data before returning
    if embeddings and onboarding_ids and names and purposes:
        return embeddings, onboarding_ids, names, purposes
    else:
        raise ValueError("No valid onboarding face data found in the database.")
