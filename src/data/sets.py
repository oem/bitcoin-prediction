def calc_training_size(length, batch_size, test_percent):
    length *= 1 - test_percent
    return(int(length - (length % batch_size)))


def calc_test_size(length, batch_size, time_steps, padded_training_size):
    length -= time_steps * 2
    length -= padded_training_size
    return(int(length - (length % batch_size)))
