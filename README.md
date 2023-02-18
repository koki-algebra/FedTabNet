# TabNet for Federated Learning
卒業論文「表形式データに対する半教師あり連合学習の検討」の実験コード<br>
[TabNet: Attentive Interpretable Tabular Learning](https://arxiv.org/abs/1908.07442v5)

# Get Started
[Docker](https://www.docker.com/) のコンテナ上で実験を行うため, 以下の手順に従ってコンテナを起動すれば実験環境は自動的に構築される.

## Preliminaries
以下のサイトに従って Docker をインストールする.<br>
https://docs.docker.com/engine/install/

## for VSCode User
[Visual Studio Code](https://code.visualstudio.com/) を使用しているユーザーは [Remote Development](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.vscode-remote-extensionpack) という拡張機能を用いるのが簡単である. 拡張機能をインストール後, コマンドパレット(command + shift + P)から "Dev Containers: Reopen in Container" を実行. Nvidia の GPU がないデバイスで実行する際は .devcontainer/devcontainer.json の
```
"runArgs": ["--gpus", "all"],
```
の行をコメントアウトしてから実行する.

## Otherwise
VSCode 以外のユーザーは Makefile を用いて起動することができる.
```
$ make up
```
を実行すると Docker Compose によってコンテナが起動される. 起動したコンテナには
```
$ make attach
```
で入れる. コンテナを終了する場合は
```
$ make down
```
を実行する.

# Run Experiments
```
src/tabnet.py
```
を実行すると通常の中央集権型の TabNet が実行される. 使用するデータセットやハイパーパラメータを指定するにはコードを編集すれば良い.
```
src/fedtabnet_pretrain.py
src/fedtabnet_fine_tuning.py
```
はそれぞれ FedTabNet の Pre-training と Fine-tuning を実行する. FedTabNet の設定は以下のディレクトリに yaml ファイルを作って記述する.
```
config/
```
FedTabNet は [EasyFL](https://github.com/EasyFL-AI/EasyFL) という Federated Learning のライブラリを用いており, yaml ファイルの詳細は [ドキュメント](https://easyfl.readthedocs.io/en/latest/)を参照されたい.<br>
```
src/fedtabnetpl.py
```
を実行すると FedTabNetPL の Fine-tuning が実行される. Pre-training は fedtabnet_pretraining.py を用いれば良い.