class CountVectorizer():
    def __init__(self):
        self.bag_of_words = []

    def get_list_of_words(self, text: str):
        return (x.lower() for x in text.split())

    def get_unique_words(self, text):
        seen = set()
        unique = []
        for each in self.get_list_of_words(text):
            if each not in seen:
                seen.add(each)
                unique.append(each)
        return unique

    def get_feature_names(self, texts: list) -> list:
        res = []
        for text in texts:
            res = res + [k for k in self.get_unique_words(text) if k not in res]
        self.bag_of_words = res
        return list(self.bag_of_words)

    def count_vector(self, text: str):
        v = [0] * len(self.bag_of_words)
        for each in enumerate(self.bag_of_words):
            for word in self.get_list_of_words(text):
                if each[1] == word:
                    v[each[0]] += 1
        return v

    def fit_transform(self, texts: list):
        self.get_feature_names(texts)
        return [self.count_vector(text) for text in texts]


if __name__ == '__main__':
    corvus = ['Crock Pot Pasta Never boil pasta again',
              'Pasta Pomodoro Fresh ingredients Parmesan to taste',
              'pasta pasta pasta pasta']
    vectorizer = CountVectorizer()
    print(vectorizer.get_feature_names(corvus))
    print(vectorizer.fit_transform(corvus))
