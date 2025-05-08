import os
from seal import *
from uni_henn.constants import *

KEYS_DIR = "/home/user/Desktop/uni-henn/keys"  # Change this to your preferred location

class Context:
    def _get_params(self, poly_modulus_degree, coeff_modulus):
        """Get default encryption parameters"""
        params = EncryptionParameters(scheme_type.ckks)
        params.set_poly_modulus_degree(poly_modulus_degree)
        params.set_coeff_modulus(
            CoeffModulus.Create(poly_modulus_degree, coeff_modulus)
        )
        return params

    def _generate_keys(self, context):
        """Generate encryption keys"""
        keygen = KeyGenerator(context)
        self.public_key = keygen.create_public_key()
        self.secret_key = keygen.secret_key()
        self.galois_key = keygen.create_galois_keys()
        self.relin_keys = keygen.create_relin_keys()

    def _save_keys(self, key_dir):
        """Save keys to the specified directory"""
        os.makedirs(key_dir, exist_ok=True)
        self.secret_key.save(os.path.join(key_dir, "secret.key"))
        self.public_key.save(os.path.join(key_dir, "public.key"))
        self.relin_keys.save(os.path.join(key_dir, "relin.key"))
        self.galois_key.save(os.path.join(key_dir, "galois.key"))
        print(f"🔐 Keys saved in {key_dir}")

    def _load_keys(self, key_dir, context):
        """Load keys from the specified directory"""
        if not all(os.path.exists(os.path.join(key_dir, f"{name}.key")) 
                   for name in ["secret", "public", "relin", "galois"]):
            return False  # Some keys are missing
        
        self.secret_key = SecretKey()
        self.secret_key.load(context, os.path.join(key_dir, "secret.key"))
        self.public_key = PublicKey()
        self.public_key.load(context, os.path.join(key_dir, "public.key"))
        self.relin_keys = RelinKeys()
        self.relin_keys.load(context, os.path.join(key_dir, "relin.key"))
        self.galois_key = GaloisKeys()
        self.galois_key.load(context, os.path.join(key_dir, "galois.key"))
        
        print(f"🔑 Keys loaded from {key_dir}")
        return True

    def __init__(self, N=NUMBER_OF_SLOTS, depth=DEPTH, LogQ=FRACTION_SCALE, LogP=INTEGER_SCALE + FRACTION_SCALE, scale=0):
        """Initialize SEAL context and manage keys"""
        coeff_modulus = [LogP] + [LogQ] * depth + [LogP]
        context = SEALContext(self._get_params(N * 2, coeff_modulus))

        self.number_of_slots = N
        self.depth = depth
        self.scale = scale if scale != 0 else 2**LogQ

        if not self._load_keys(KEYS_DIR, context):  # Try loading keys
            print("⚠️ Keys not found, generating new ones...")
            self._generate_keys(context)
            self._save_keys(KEYS_DIR)  # Save them after generation

        self.encoder = CKKSEncoder(context)
        self.encryptor = Encryptor(context, self.public_key)
        self.evaluator = Evaluator(context)
        self.decryptor = Decryptor(context, self.secret_key)

