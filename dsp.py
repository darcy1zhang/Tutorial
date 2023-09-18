import numpy as np
import matplotlib.pyplot as plt

def perform_ifft(signal, show):
    """
    Description:
        Perform Inverse Fast Fourier Transform (IFFT) on a given FFT result.

    Params:
        fft_result (numpy.ndarray): The FFT result.

    Return:
        numpy.ndarray: Array of complex numbers representing the IFFT result.
    """

    ifft_result = np.fft.ifft(signal)
    if show:
        plt.plot(ifft_result)
        plt.show()

    return ifft_result

def plot_scg(signal):
    plt.plot(signal)
    plt.show()
    return signal

def perform_fft(signal, sampling_rate, show):
    """
    Description:
        Perform Fast Fourier Transform (FFT) on a given signal.

    Params:
        signal (numpy.ndarray): The input signal.
        sampling_rate (float): The sampling rate of the signal.

    Return:
        numpy.ndarray: Array of complex numbers representing the FFT result.
        numpy.ndarray: Array of corresponding frequency values.
    """

    n = len(signal)
    frequencies = np.fft.fftfreq(n, d=1 / sampling_rate)
    fft_result = np.fft.fft(signal)

    if show:
        plt.figure(figsize=(8, 6))
        plt.plot(frequencies, np.abs(fft_result))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Magnitude')
        plt.title('FFT Spectrum')
        plt.grid()
        plt.show()

    return fft_result


if __name__ == '__main__':
    import Academic_signals_generation
    import Feature_extraction_and_analysis

    def check_arguments():
        pass

    def get_params(func_sequence, pattern = r'\((.*?)\)'):
        import re

        params = {}
        # for input_str in func_sequence.split():
        for input_str in func_sequence:
            print(input_str)
            match = re.search(pattern, input_str)
            if match:
                parameters_str = match.group(1)
                parameters_list = parameters_str.split(',')
                parameters_list = [param.strip() for param in parameters_list]
                if parameters_list == ['']:
                    parameters_dic = None
                else:
                    parameters_dic = {}
                    for paramters in parameters_list:
                        check_arguments()
                        paramters_splits = paramters.split('=')
                        if paramters_splits[0] == 'signal':
                            continue
                        if paramters_splits[0] == 'show':
                            parameters_dic[paramters_splits[0]] = True if paramters_splits[-1] == 'true' else False
                            continue
                        # print(paramters_splits[-1])
                        parameters_dic[paramters_splits[0]] = float(paramters_splits[-1])
                params[input_str.split('(')[0]] = parameters_dic
            else:
                params[input_str] = None
        return params

    def check_data_source(data_source):
        data_sources = ['scg']
        if data_source in data_sources:
            return True
        else:
            print(data_source, data_sources)
            return False

    def check_callable(func_name):
        func = globals()[func_name]
        if func is not None and callable(func):
            return True
        else:
            print(f"函数 '{func_name}' 未找到或不可调用。")
            print(f"")
            return False

    def load_data(data_source):
        # it is an easy implement
        signals_train, labels_train, duration, fs = Academic_signals_generation.load_scg(0.1, 'train')
        return signals_train[1]

    def check_func_types(funcname_params):
        transformers= []
        estimators = []
        pass


    while True:

        inputs = input("Enter Your Command: ").lower().split(' ')

        # quit
        if inputs[0] in ['q', 'quit']:
            print('quit')
            break
        if 'q' in inputs or 'quit' in inputs:
            print('quit')
            break

        # help
        if inputs[0] in ['h', 'help']:
            for i in range(1, len(inputs)):
                func_name = inputs[i]
                print(help(globals()[func_name]))
            continue

        if inputs[0] in ['s', 'save']:
            pass

        middle_res = None

        if not check_data_source(inputs[0]):
            print(inputs[0])
            print('data source error')
            continue
            # break
        else:
            data_source = inputs[0]
            middle_res = load_data(data_source)

        funcname_params = get_params(inputs[1:])

        # if not check_func_types(funcname_params):
        #     print('func type error')
        #     continue
            # break

        # iterate dic
        for func_name, params in funcname_params.items():
            """
            # 1. def data_source = ['scg'] and check data source
            # 2. get {func_name->str, {parameter_1->str: para_1->str / number}} & check if all parameters follow the Key Word arguments
            # 3. check if func_name is callable
            # 4. check if all parameters as same as the functions' parameters name
            6. check if the next function can accept the output of the previous function as the input
            """

            if not check_callable(func_name):
                break
            print(f'Function Name: {func_name}')
            func = globals()[func_name]
            params['signal'] = middle_res
            print(f'Input shape of Function: {middle_res.shape}')
            middle_res = func(**params)
            print(f'Output shape of Function: {middle_res.shape}')
            print()