from typing import Tuple

from make87 import (
    initialize,
    get_subscriber_topic,
    get_publisher_topic,
    resolve_topic_name,
    MultiSubscriberTopic,
)
from make87_messages.geometry.box.box_2d_pb2 import Box2DAxisAligned
from make87_messages.image.compressed.image_jpeg_pb2 import ImageJPEG
import cv2
import numpy as np


def main():
    initialize()
    topic_name = resolve_topic_name("CROPPED_IMAGE")
    publisher_topic = get_publisher_topic(name=topic_name, message_type=ImageJPEG)

    def crop_and_publish_image(messages: Tuple[Box2DAxisAligned, ImageJPEG]):
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

    bounding_box_topic = get_subscriber_topic(name=resolve_topic_name("BOUNDING_BOX_2D"), message_type=Box2DAxisAligned)
    image_topic = get_subscriber_topic(name=resolve_topic_name("IMAGE_DATA"), message_type=ImageJPEG)

    multi_subscriber = MultiSubscriberTopic(delta_time=0.1)
    multi_subscriber.add_topic(bounding_box_topic)
    multi_subscriber.add_topic(image_topic)
    multi_subscriber.subscribe(crop_and_publish_image)


if __name__ == "__main__":
    main()
