const form = document.querySelector('#search-form');
const input = document.querySelector('#query');
const results = document.querySelector('#results');
const status = document.querySelector('#status');
const title = document.querySelector('#result-title');
const count = document.querySelector('#result-count');
const template = document.querySelector('#result-template');

let activeRequest = null;

function setLoading(query) {
  results.replaceChildren();
  title.textContent = `Đang tìm “${query}”`;
  status.textContent = 'Đang truy vấn cơ sở dữ liệu RxNorm…';
  status.hidden = false;
  count.hidden = true;
}

function render(data) {
  results.replaceChildren();
  title.textContent = data.query ? `Kết quả cho “${data.query}”` : 'Nhập từ khóa để bắt đầu';
  count.textContent = `${data.count} kết quả`;
  count.hidden = false;

  if (!data.results.length) {
    status.textContent = 'Không tìm thấy thuốc phù hợp. Hãy thử tên hoạt chất, biệt dược khác hoặc RxCUI.';
    status.hidden = false;
    return;
  }
  status.hidden = true;

  data.results.forEach((drug, index) => {
    const node = template.content.cloneNode(true);
    const card = node.querySelector('.drug-card');
    card.style.animationDelay = `${Math.min(index * 25, 180)}ms`;
    node.querySelector('h3').textContent = drug.name;
    node.querySelector('.match').textContent = drug.match;
    node.querySelector('.rxcui').textContent = drug.rxcui;
    node.querySelector('.tty').textContent = drug.tty;
    node.querySelector('.state').textContent = drug.suppressed ? 'Suppressed' : 'Đang hoạt động';

    const list = node.querySelector('.synonyms');
    drug.names.forEach(item => {
      const li = document.createElement('li');
      const name = document.createElement('span');
      const tty = document.createElement('small');
      name.textContent = item.name;
      tty.textContent = item.tty;
      li.append(name, tty);
      list.append(li);
    });

    const copy = node.querySelector('.copy');
    copy.addEventListener('click', async () => {
      await navigator.clipboard.writeText(drug.rxcui);
      copy.textContent = 'Đã chép';
      setTimeout(() => { copy.textContent = 'Sao chép'; }, 1200);
    });
    results.append(node);
  });
}

async function search(query) {
  query = query.trim();
  if (!query) {
    render({ query: '', count: 0, results: [] });
    return;
  }
  activeRequest?.abort();
  activeRequest = new AbortController();
  setLoading(query);
  try {
    const response = await fetch(`api/search?q=${encodeURIComponent(query)}&limit=20`, {
      signal: activeRequest.signal,
    });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    render(await response.json());
  } catch (error) {
    if (error.name === 'AbortError') return;
    status.hidden = false;
    status.classList.add('error');
    status.textContent = 'Không thể kết nối tới database. Hãy kiểm tra server Python.';
  }
}

form.addEventListener('submit', event => {
  event.preventDefault();
  search(input.value);
});

document.querySelectorAll('[data-query]').forEach(button => {
  button.addEventListener('click', () => {
    input.value = button.dataset.query;
    search(input.value);
  });
});
