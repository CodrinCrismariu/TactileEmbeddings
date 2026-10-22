"""Launch the local EgoTouch reconstruction viewer."""
from tactilelatent.autoencoder_viewer import main

if __name__ == "__main__":
    from tactilelatent.acquisition_joint_viewer import install
    install()
    from tactilelatent.bilateral_runtime import install_viewer_loaders
    install_viewer_loaders()
    from tactilelatent.training_dashboard import install as install_training_dashboard
    install_training_dashboard()
    main()
