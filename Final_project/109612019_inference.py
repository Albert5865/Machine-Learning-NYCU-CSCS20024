from fastai.vision.all import*

path = Path('./training/data')

def generate(test_folder, output_csv='predictions.csv', model_path="./training/data/model.pkl"):

    learn = load_learner(model_path)
    test_dl = learn.dls.test_dl(get_image_files(path/'test'))
    preds, _ = learn.get_preds(dl=test_dl)

    # Assuming you have a list of test filenames
    test_filenames = get_image_files(path/'test')
    ids = [fname.stem for fname in test_filenames]

    # Make predictions using your model (replace this with your actual prediction code)
    test_dl = learn.dls.test_dl(test_filenames)
    preds, _ = learn.get_preds(dl=test_dl)

    # Get the predicted labels using the vocabulary
    predicted_labels = [learn.dls.vocab[i] for i in torch.argmax(preds, dim=1)]

    # Create a DataFrame for submission
    submission_df = pd.DataFrame({'id': ids, 'label': predicted_labels})

    # Save the DataFrame to a CSV file
    submission_df.to_csv('submission.csv', index=False)

if __name__ == '__main__':
    generate(path/'test')