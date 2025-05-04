from matplotlib.backends.backend_pdf import PdfPages

def generate_report(model, test_images, output_path="report.pdf"):
    with PdfPages(output_path) as pdf:
        # 1. Model architecture diagram
        plt.figure(figsize=(12, 8))
        plot_model(model, to_file="model.png", show_shapes=True)
        pdf.savefig()
        
        # 2. Sample predictions
        for img in test_images[:5]:
            pred = model.predict(img[np.newaxis, ...])
            plt.figure()
            plt.imshow(img, cmap='gray')
            plt.title(f"Predicted class: {np.argmax(pred)}")
            pdf.savefig()