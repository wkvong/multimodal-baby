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
        if isinstance(timestamp, str):
            timestamp_one = msplit(timestamp, '-')[0]
         
            if timestamp_one != '':
                splits = msplit(timestamp_one, (':', '.', ',', ';'))
         
                if splits[0] == '':
                    splits[0] = '0'
         
                if len(splits) == 1:
                    splits.append('0')
                else:
                    # sometimes only the tens of seconds are reported as single digits
                    # this converts this correctly
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
         
                timestamp_one_secs = int(splits[0]) * 60 + int(splits[1])
                if timestamp_one_secs > 2000:
                    print(f'timestamp out of range: {timestamp_one_secs}')
                    timestamp_one_secs = np.nan
         
                new_timestamps.append(timestamp_one_secs)
            else:
                new_timestamps.append(None)  # handles non-empty string that is not a timestamp
        else:
            new_timestamps.append(None)  # handles non-strings like nans

    return new_timestamps

