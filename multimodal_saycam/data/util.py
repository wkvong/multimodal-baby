import numpy as np


def msplit(string, delimiters):
    """Split with multiple delimiters."""
    delimiters = tuple(delimiters)
    stack = [string, ]

    for delimiter in delimiters:
        for i, substring in enumerate(stack):
            substack = substring.split(delimiter)
            stack.pop(i)
            for j, _substring in enumerate(substack):
                stack.insert(i + j, _substring)

    return stack


def convert_timestamps_to_seconds(timestamps):
    """Function to convert a variety of starting timestamps from SAYCam transcripts into seconds."""
    
    new_timestamps = []
    for timestamp in timestamps:
        timestamp = str(timestamp)  # convert to string
        if timestamp != 'nan':
            timestamp_one = msplit(timestamp, '-')[0]  # get starting timestamp
         
            if timestamp_one != '':
                splits = msplit(timestamp_one, (':', '.', ',', ';'))
         
                if splits[0] == '':
                    splits[0] = '0'
         
                if len(splits) == 1:
                    splits.append('0')
                else:
                    # sometimes only the tens of seconds are reported as single digits
                    # this converts these values to seconds
                    if splits[1] == '1':
                        splits[1] = '10'
                    elif splits[1] == '2':
                        splits[1] = '20'
                    elif splits[1] == '3':
                        splits[1] = '30'
                    elif splits[1] == '4':
                        splits[1] = '40'
                    elif splits[1] == '5':
                        splits[1] = '50'

                # trim whitespace
                splits[0] = splits[0].strip()
                splits[1] = splits[1].strip()

                if len(splits[1]) <= 2:
                    # handle proper timestamps
                    timestamp_one_secs = int(splits[0]) * 60 + int(splits[1])
                else:
                    # handle float-like timestamps
                    # TODO: figure out what the floats encode, otherwise
                    # for now just setting them to None
                    timestamp_one_secs = None
         
                new_timestamps.append(timestamp_one_secs)
            else:
                new_timestamps.append(None)  # handles non-empty string that is not a timestamp
        else:
            new_timestamps.append(None)  # handles non-strings like nans

    return new_timestamps
