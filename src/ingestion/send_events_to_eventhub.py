import json
import os
from dotenv import load_dotenv
from azure.eventhub import EventHubProducerClient, EventData

from src.ingestion.event_producer_simulator import generate_events

def main():
    load_dotenv()

    conn_str = os.getenv("EVENT_HUB_CONNECTION_STRING")
    hub_name = os.getenv("EVENT_HUB_NAME")

    if not conn_str or not hub_name:
        raise ValueError("Missing EVENT_HUB_CONNECTION_STRING or EVENT_HUB_NAME in .env")

    producer = EventHubProducerClient.from_connection_string(
        conn_str=conn_str,
        eventhub_name=hub_name
    )

    events = generate_events(n_users=50, n_events=200)

    with producer:
        batch = producer.create_batch()
        for e in events:
            payload = json.dumps(e)
            try:
                batch.add(EventData(payload))
            except ValueError:
                producer.send_batch(batch)
                batch = producer.create_batch()
                batch.add(EventData(payload))
        if len(batch) > 0:
            producer.send_batch(batch)

    print(f"✅ Sent {len(events)} events to Event Hubs: {hub_name}")

if __name__ == "__main__":
    main()
