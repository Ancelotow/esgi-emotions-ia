class Model:
    def __init__(self, emotion, img):
        self.emotion = emotion
        self.data = []
        for x in range(img.width):
            for y in range(img.height):
                self.data.append(img.getpixel((x, y)))

    def to_array(self):
        arr = self.data.copy()
        # arr.append(self.emotion)
        return arr