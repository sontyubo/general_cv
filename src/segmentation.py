import torch
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

# 学習済みのモデルをロード
model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)
model.eval()

# 画像をロードしてテンソル型に変換する前処理を実行
def preprocess_image(img_path):
    input_image = Image.open(img_path)
    preprocess = T.Compose([
        T.ToTensor()
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # バッチ次元を追加
    return input_batch

# セグメンテーションの実行
def segment_image(img_path):
    # 画像の前処理を実行
    input_batch = preprocess_image(img_path)
    # pytorchがGPUの情報を確認する
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    with torch.no_grad():
        # どのクラスに属するかを判断する確率
        output = model(input_batch)['out'][0]
    # 各ピクセルがどのクラスに属するのか
    output_predictions = output.argmax(0)
    # output_prediction = 多重配列
    return output_predictions

# 結果を表示
def display_segmentation(img_path, output_predictions):
    input_image = Image.open(img_path)
    # RGBAの画像データまたは、2次元のスカラーデータを画像として表示する
    plt.imshow(input_image)
    # テンソルをCPUに移動させる
    output_predictions_cpu = output_predictions.cpu()
    plt.imshow(output_predictions_cpu, alpha=0.7)
    plt.axis('off')
    #plt.show()
    plt.savefig('image/segmentation/output.png')
    plt.close()



def main():
    # 画像パス
    img_path = 'image/segmentation/input.jpg' # ここに画像のパスを入力

    # セグメンテーションの実行と表示
    output_predictions = segment_image(img_path)
    display_segmentation(img_path, output_predictions)

    print('--- fin ---')


if __name__ == '__main__':
    main()

