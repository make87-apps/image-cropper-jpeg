from make87 import get_topic, topic_names, PublisherTopic, SubscriberTopic, MultiSubscriberTopic, MessageMetadata
from make87_messages.geometry.BoundingBox2D_pb2 import AxisAlignedBoundingBox2DFloat
from make87_messages.image.ImageJPEG_pb2 import ImageJPEG
import cv2
import numpy as np


def main():
    publisher_topic = get_topic(name=topic_names.CROPPED_IMAGE)

    def crop_and_publish_image(messages: AxisAlignedBoundingBox2DFloat, metadata: MessageMetadata):
        bounding_box_msg, image_msg = messages

        # Decode the image
        image = np.frombuffer(image_msg.data, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        # Get bounding box coordinates
        x, y = int(bounding_box_msg.x), int(bounding_box_msg.y)
        width, height = int(bounding_box_msg.width), int(bounding_box_msg.height)

        # Crop the image
        cropped_image = image[y : y + height, x : x + width]

        # Encode cropped image back to JPEG
        if cropped_image.size > 0:  # Ensure there is content to encode
            _, buffer = cv2.imencode(".jpeg", cropped_image)
            cropped_image_data = buffer.tobytes()

            # Create a new protobuf message
            cropped_image_message = ImageJPEG(data=cropped_image_data)
            publisher_topic.publish(cropped_image_message)
            print("Published cropped image")

    bounding_box_topic = get_topic(name=topic_names.BOUNDING_BOX_2D)
    image_topic = get_topic(name=topic_names.IMAGE_DATA)

    multi_subscriber = MultiSubscriberTopic(topics=[bounding_box_topic, image_topic], delta=0.1)
    multi_subscriber.subscribe(crop_and_publish_image)


if __name__ == "__main__":
    main()
