import numpy as np
import joblib
from scipy.sparse import hstack

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from config import BOT_TOKEN


ARTIFACTS_PATH = "./artifacts/"

tfidf_char = joblib.load(ARTIFACTS_PATH + "tfidf_char.pkl")
tfidf_word = joblib.load(ARTIFACTS_PATH + "tfidf_word.pkl")
models_info = joblib.load(ARTIFACTS_PATH + "models.pkl")
i2l = joblib.load(ARTIFACTS_PATH + "i2l.pkl")
weights = joblib.load(ARTIFACTS_PATH + "ensemble_weights.pkl")
code2name = joblib.load(ARTIFACTS_PATH + "code2name.pkl")

models = [m["model"] for m in models_info]
NUM_CLASSES = len(i2l)


def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


def get_proba(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return softmax(model.decision_function(X))


def rank_ensemble(models, weights, X, num_classes):
    scores = np.zeros((X.shape[0], num_classes))

    for w, model in zip(weights, models):
        probs = get_proba(model, X)
        ranks = np.argsort(-probs, axis=1)

        for i in range(X.shape[0]):
            for r, cls in enumerate(ranks[i]):
                scores[i, cls] += w / (r + 1)

    return scores


def predict_codes(text, top_k=3):
    Xc = tfidf_char.transform([text])
    Xw = tfidf_word.transform([text])
    X = hstack([Xc, Xw])

    scores = rank_ensemble(models, weights, X, NUM_CLASSES)[0]
    top_idx = np.argsort(scores)[::-1][:top_k]

    return [i2l[i] for i in top_idx]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Введите описание симптомов, и я подберу ТОП-3 возможных диагноза по классификации МКБ-10."
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text.strip()

    if len(text) < 5:
        await update.message.reply_text("Описание симптомов слишком короткое.")
        return

    codes = predict_codes(text)

    reply = "ТОП-3 возможных диагноза:\n\n"
    for i, code in enumerate(codes, 1):
        name = code2name.get(code, "Название не найдено")
        reply += f"{i}. {code} — {name}\n"

    reply += "\nРезультат носит справочный характер и не является медицинским заключением."

    await update.message.reply_text(reply)


def main():
    app = ApplicationBuilder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    print("Bot is running...")
    app.run_polling()


if __name__ == "__main__":
    main()
