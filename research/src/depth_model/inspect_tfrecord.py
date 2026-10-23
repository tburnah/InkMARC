import sys
import tensorflow as tf

def inspect_tfrecord(path, max_records=5):
    print("Opening:", path)

    dataset = tf.data.TFRecordDataset(path)

    count = 0

    for raw_record in dataset:
        count += 1

        if count <= max_records:
            print("\n===== RECORD", count, "=====")

            example = tf.train.Example()
            example.ParseFromString(raw_record.numpy())

            for k, v in example.features.feature.items():
                kind = v.WhichOneof("kind")

                if kind == "bytes_list":
                    size = len(v.bytes_list.value[0])
                    print(f"{k}: bytes ({size} bytes)")

                elif kind == "float_list":
                    print(f"{k}: float_list len={len(v.float_list.value)}")

                elif kind == "int64_list":
                    print(f"{k}: int64_list len={len(v.int64_list.value)}")

                else:
                    print(f"{k}: unknown")

        if count % 1000 == 0:
            print("Read", count, "records")

    print("\nTOTAL RECORDS:", count)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python inspect_tfrecord.py file.tfrecord")
        sys.exit(1)

    inspect_tfrecord(sys.argv[1])