import re

with open('encoder_bipartite.py', 'r') as f:
    content = f.read()

# Replace the problematic _hash_ids method
old_pattern = r'    @staticmethod\n    def _hash_ids\(self, ids: torch\.Tensor, buckets: int\) -> torch\.Tensor:\n        # ids: \(...\,\) int64/long\n        # Use multiplicative hashing for stability\.\n        # \(Knuth'"'"'s multiplicative method\)\n        x = ids\.to\(torch\.int64\)\n        x = torch\.fmod\(x \* 2654435761, 2\*\*64\)\n        return \(x % buckets\)\.to\(torch\.long\)\n'

new_code = '''    @staticmethod
    def _hash_ids(ids: torch.Tensor, buckets: int) -> torch.Tensor:
        # ids: (...,) int64/long
        # Use multiplicative hashing for stability.
        # (Knuth's multiplicative method) - Safe version to avoid overflow
        x = ids.to(torch.float64)
        x = torch.fmod(x * 2654435761.0, float(2**64))
        return (x.long() % buckets).to(torch.long)
'''

content = re.sub(old_pattern, new_code, content, flags=re.DOTALL)

with open('encoder_bipartite.py', 'w') as f:
    f.write(content)

print("Fixed!")
