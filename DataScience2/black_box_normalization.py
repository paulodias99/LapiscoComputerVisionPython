import matplotlib.pyplot as plt

class BlackBox():
    def __init__(self, chain_codes, signals): #sinais e os codigos das imagens
        self.chain_codes = chain_codes
        self.signals = signals
        self.black_box_normalization()
    def black_box_normalization(self):
        self.new_chain_codes = []
        self.smaller = 9999
        for value in self.signals:
            if value < self.smaller:
                self.smaller = value

        print('the smaller', self.smaller)

        for i in range(len(self.signals)):
            proportion = self.signals[i] / self.smaller
            print(proportion)

            if (self.signals[i] == self.smaller):
                self.new_chain_codes.append(self.chain_codes[i])
                plt.plot(self.chain_codes[i])
                plt.show()
            else:
                aux = []
                idx = 0
                rest = 0
                while (idx < self.signals[i]):
                    aux.append(self.chain_codes[i][idx])
                    rest = (rest + proportion) % 1
                    idx = int((idx + proportion + rest)//1)

                self.new_chain_codes.append(aux)
                plt.plot(aux)
                plt.show()

        for i in range(len(self.new_chain_codes)):
            if (len(self.new_chain_codes[i]) < self.smaller):
                self.new_chain_codes[i].append(self.chain_codes[i][self.signals[i] - 1])
        return self.new_chain_codes #função me retorna os novos codigos